import os

import torch
import torch.nn as nn

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize
# Any POOLING other than 'voxelize' names a method package at the repo root (e.g. qnpruner/); see method_registry.py.
from .method_registry import is_external, load_method
from torch_scatter import scatter_mean
from .position_encodings import PositionEmbeddingLearnedMLP


class PromptEncoder(nn.Module):

    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)

    def forward(self, clicks):
        pos_embed = self.encode_pe(clicks)
        return pos_embed


class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = os.environ.get('POOLING', 'voxelize')
        self.voxel_size = float(os.environ.get('VOXEL_SIZE', '0.2'))
        # K for a token-selection method (POOLING=qnpruner); unused by 'voxelize'
        self.target_tokens = int(os.environ.get('TARGET_TOKENS', '0'))
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        print(f"[VideoTower] num_frames={self.num_frames}", flush=True)
        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        self.is_loaded = True

    def forward(self, features, depths, poses, intrinsics, lengths=None, projector=None, query_texts=None):
        """
        Args:
            - features: (B, V, 1024, 24, 24), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4), camera-to-world poses
            - intrinsics: (B, V, 4, 4), camera intrinsics
            - lengths: (B,), view count per scene
            - projector: the model's ``mm_projector`` (read by QNPruner for its relevance scores)
            - query_texts: list of B question strings (the raw user query); read by QNPruner only

        Returns:
            - pooled_video_features: (Bn, 1024)
            - batch_offset: (B,) int32
        """
        B, V, C, H, W = features.shape
        _log = not getattr(self, '_logged_once', False)
        assert intrinsics.dim() == 4

        # Depth backprojection -> 3D world coordinates (axis-aligned ScanNet frame, see video_processor)
        feat_xyz, _dense_xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # Spatial-aware 3D position embedding
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_xyz = feat_xyz.reshape(B, V * H * W, 3)
        visual_2d_features = features.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)
        if lengths is not None:
            lengths = lengths * H * W

        if self.pooling == 'voxelize':
            # LLaVA-3D: one token per voxel (scatter_mean over the voxel grid), no selection
            p2v = voxelize(feat_xyz, self.voxel_size)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(B)])
            xyz_list = [scatter_mean(video_xyz[b], p2v[b], dim=0) for b in range(B)]
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)

        elif is_external(self.pooling):
            # A token-selection package at the repo root (POOLING=qnpruner). It starts from the same voxel tokens
            # and returns the same triple as the branch above.
            ctx = dict(
                B=B, V=V, C=C, H=H, W=W,
                features=features, feat_xyz=feat_xyz,
                video_features=video_features, video_xyz=video_xyz,
                visual_2d_features=visual_2d_features, projector=projector,
                query_texts=query_texts, log=_log,
            )
            pooled_video_features, xyz_list, batch_offset = load_method(self.pooling).compress(self, ctx)

        else:
            raise NotImplementedError(f"Unknown pooling: '{self.pooling}'")

        self._voxel_xyz = xyz_list
        self._logged_once = True
        return pooled_video_features, batch_offset

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
