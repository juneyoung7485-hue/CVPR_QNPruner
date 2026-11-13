"""Shared building blocks for query-guided compression methods.

Nothing here runs unless a method package calls it. What a version does with
the relevance (DPP quality, top-k, merge weights, ...) belongs in that
version's own package, not here.

* ``QueryEncoder`` -- frozen CLIP text tower + ``text_projection`` +
  ``visual_projection`` of the checkpoint LLaVA-3D's vision tower uses. The
  vision transformer itself is already on the GPU as LLaVA-3D's tower, so a
  second copy is never kept.
* ``voxel_query_similarity`` -- per-voxel cosine between the question and
  CLIP's semantic patch tokens (``CLIPVisionTower._query_patch_features``:
  last layer + post_layernorm), pooled onto a voxel grid.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


def env_flag(name, default='0'):
    return os.environ.get(name, default) not in ('0', 'false', 'False', '')


class QueryEncoder(nn.Module):
    """``CLIPModel`` is loaded on the CPU once, the text tower and projections
    are lifted out, and the vision half is freed before anything moves to the
    GPU. All parameters frozen, eval mode."""

    def __init__(self, clip_name, device=None, dtype=torch.float32):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.clip_name = clip_name
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        full = CLIPModel.from_pretrained(clip_name, torch_dtype=torch.float32)
        self.text_model = full.text_model
        self.text_projection = full.text_projection
        self.visual_projection = full.visual_projection
        del full   # drops the duplicate vision transformer

        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        if device is not None:
            self.to(device=device, dtype=dtype)

    @property
    def device(self):
        return self.text_projection.weight.device

    @torch.no_grad()
    def encode_text(self, texts):
        """(B,) list of str -> (B, D_clip) L2-normalized fp32 embeddings.

        CLIP's text context is 77 tokens. SQA3D's situation + question can run
        past it, so the tail is truncated rather than raising.
        """
        tok = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.tokenizer.model_max_length,
                             return_tensors='pt').to(self.device)
        pooled = self.text_model(**tok).pooler_output
        return F.normalize(self.text_projection(pooled).float(), dim=-1)

    @torch.no_grad()
    def project_visual(self, feats):
        """(N, D_vision) post-layernormed CLIP tokens -> (N, D_clip) normalized."""
        w = self.visual_projection.weight
        return F.normalize(self.visual_projection(feats.to(w.device, w.dtype)).float(), dim=-1)


_QUERY_ENCODERS = {}


def get_query_encoder(clip_name, device):
    """One lazily-built encoder per (checkpoint, device), kept outside any
    ``nn.Module`` so it never lands in the LLaVA state_dict or its ``.to()``."""
    key = (clip_name, str(device))
    if key not in _QUERY_ENCODERS:
        print(f'[QueryEncoder] loading frozen CLIP text encoder + projections from {clip_name}', flush=True)
        _QUERY_ENCODERS[key] = QueryEncoder(clip_name, device=device)
    return _QUERY_ENCODERS[key]


def normalize_query_relevance(sim, eps=1e-6):
    """Scene-internal min-max: (n,) cosine -> (n,) in [0, 1]."""
    sim = sim.float()
    return (sim - sim.min()) / (sim.max() - sim.min() + eps)


def voxel_query_similarity(query_patch_features, p2v, n_vox, text_emb, encoder):
    """Per-voxel CLIP cosine similarity to the question.

    ``query_patch_features`` (N, D_vision) must be flattened in the same
    (V, H, W) order as the tokens ``p2v`` (N,) indexes.
    Returns (sim (n_vox,), vox_semantic (n_vox, D_vision)).
    """
    assert query_patch_features.shape[0] == p2v.shape[0], (
        f'query patch features ({query_patch_features.shape[0]}) and p2v '
        f'({p2v.shape[0]}) index different token sequences')
    vox_semantic = scatter_mean(query_patch_features.float(), p2v, dim=0, dim_size=n_vox)
    vox_query_feat = encoder.project_visual(vox_semantic)            # (n_vox, D_clip)
    q = F.normalize(text_emb.float().to(vox_query_feat.device), dim=-1)
    return vox_query_feat @ q, vox_semantic
