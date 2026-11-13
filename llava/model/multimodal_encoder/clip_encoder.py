import os
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self._vz_metric = None
        self._vz_metric_buffer = None
        # VisionZip and ToSA both merge on CLIP layer -2 key states.
        if os.environ.get('POOLING', 'voxelize') in ('visionzip', 'tosa'):
            self._install_visionzip_metric_hook()

        # Query-guided methods read CLIP's *semantic* patch tokens (last layer +
        # the final post_layernorm, i.e. the space visual_projection maps into
        # the joint image-text space) off this same forward. Kept only when the
        # method package asks (NEEDS_QUERY_PATCH_FEATURES), so every other
        # pooling runs exactly as before.
        from .method_registry import method_flag
        self._query_patch_features = None
        self.store_query_patch_features = method_flag(
            os.environ.get('POOLING', 'voxelize'), 'NEEDS_QUERY_PATCH_FEATURES')
        # Attention each patch *receives* from the other patches (same layer as
        # the CLS row, head mean then query-patch mean). Kept only when the
        # method package asks (NEEDS_PATCH_ATTN), e.g. query_token_compression_v3_2.
        self._patch_attn = None
        self.store_patch_attn = method_flag(
            os.environ.get('POOLING', 'voxelize'), 'NEEDS_PATCH_ATTN')

        self.is_loaded = True

    def _install_visionzip_metric_hook(self):
        """Capture the head-averaged key states VisionZip and ToSA merge on.

        Both upstreams reach for the same tensor and both get it by replacing
        attention code: VisionZip monkey-patches ``CLIPAttention.forward`` to
        also return ``self._shape(k_proj(x), -1, bsz).mean(1)``, ToSA rewrites
        SigLIP's attention to return ``key_states.mean(dim=1)``. A forward hook
        on ``k_proj`` reproduces that tensor exactly without replacing any
        transformers code, so every other pooling method keeps running on the
        stock CLIP path. The layer is the one the features and the CLS
        attention already come from (``CLS_ATTN_LAYER``, -2 by default).
        """
        layer_index = int(os.environ.get('CLS_ATTN_LAYER', '-2'))
        attn = self.vision_tower.vision_model.encoder.layers[layer_index].self_attn
        num_heads, head_dim = attn.num_heads, attn.head_dim
        captured = []

        def hook(_module, _inputs, output):
            bsz, seq_len, _ = output.shape
            captured.append(
                output.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).mean(1))

        attn.k_proj.register_forward_hook(hook)
        self._vz_metric_buffer = captured
        print(f"[VisionZip] key metric hook on CLIP layer {layer_index}", flush=True)

    def _collect_visionzip_metric(self, dtype):
        """Drain the hook buffer into ``_vz_metric`` as [B, num_patches, head_dim]."""
        buffer = self._vz_metric_buffer
        if not buffer:
            self._vz_metric = None
            return
        metric = buffer[0] if len(buffer) == 1 else torch.cat(buffer, dim=0)
        self._vz_metric = metric[:, 1:].to(dtype)   # drop CLS: it has no 3D position
        buffer.clear()

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # (B, n, C)
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def query_patch_select(self, image_forward_outs):
        """CLIP-joint-space-ready patch tokens: last hidden state, CLS dropped,
        post_layernorm applied. Independent of ``select_layer``, which keeps
        feeding LLaVA-3D's own visual tokens."""
        semantic_patch = image_forward_outs.last_hidden_state[:, 1:]
        semantic_patch = self.vision_tower.vision_model.post_layernorm(semantic_patch)
        return semantic_patch.detach()

    @torch.no_grad()
    def forward(self, images):
        cls_attn_layer = int(os.environ.get('CLS_ATTN_LAYER', '-2'))
        if self._vz_metric_buffer is not None:
            # A forward that raised could have left captures behind; this pass
            # owns the buffer from here on.
            self._vz_metric_buffer.clear()
        if type(images) is list:
            image_features = []
            cls_attns = []
            patch_attns = []
            query_patches = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                attn = image_forward_out.attentions[cls_attn_layer]
                cls_attns.append(attn[:, :, 0, 1:].mean(dim=1))       # (1, 576)
                if self.store_patch_attn:
                    patch_attns.append(attn[:, :, 1:, 1:].mean(dim=1).mean(dim=1))   # (1, 576)
                if self.store_query_patch_features:
                    query_patches.append(self.query_patch_select(image_forward_out))
            self._cls_attn = torch.cat(cls_attns, dim=0).to(images[0].dtype)
            self._patch_attn = torch.cat(patch_attns, dim=0).to(images[0].dtype) if patch_attns else None
            self._query_patch_features = torch.cat(query_patches, dim=0) if query_patches else None
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            attn = image_forward_outs.attentions[cls_attn_layer]
            self._cls_attn = attn[:, :, 0, 1:].mean(dim=1).to(images.dtype)  # (B, 576)
            self._patch_attn = (attn[:, :, 1:, 1:].mean(dim=1).mean(dim=1).to(images.dtype)
                                if self.store_patch_attn else None)            # (B, 576)
            self._query_patch_features = (self.query_patch_select(image_forward_outs)
                                          if self.store_query_patch_features else None)  # (B, 576, 1024)

        if self._vz_metric_buffer is not None:
            self._collect_visionzip_metric(
                images[0].dtype if type(images) is list else images.dtype)

        return image_features

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

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2