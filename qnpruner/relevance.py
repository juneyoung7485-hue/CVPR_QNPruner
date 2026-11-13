"""Step 1 of QNPruner: word-voxel relevance from frozen LLaVA-3D parts only (no extra text encoder).

    e(w)      mean of the LLM input embeddings of w's sub-word tokens (special tokens removed)
    r_w(i)    cos(P(f2d_i), e(w)), P = the LLaVA-3D projector, f2d_i = voxel mean of CLIP layer -2 patch features
"""
import torch
import torch.nn.functional as F

_TOKENIZERS = {}
_WORD_EMB = {}


def tokenizer(path):
    if path not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path, use_fast=False)
    return _TOKENIZERS[path]


def word_embedding(word, embed_tokens, tok, tok_name):
    """(D,) float32 CPU tensor or None (no token left)."""
    key = (tok_name, word)
    if key not in _WORD_EMB:
        special = set(getattr(tok, 'all_special_ids', []) or [])
        raw = torch.as_tensor(tok(word, add_special_tokens=False).input_ids).reshape(-1).tolist()
        ids = [int(i) for i in raw if int(i) not in special]
        e = None
        if ids:
            with torch.no_grad():
                w = embed_tokens.weight
                e = embed_tokens(torch.tensor(ids, device=w.device)).float().mean(0).detach().cpu()
        _WORD_EMB[key] = e
    return _WORD_EMB[key]


def word_relevance(P2d, embeddings):
    """(N, J) cosine between the projected 2D voxel features (N, D) and the word embeddings (list of (D,))."""
    E = F.normalize(torch.stack(embeddings).float(), dim=1)
    return F.normalize(P2d, dim=1) @ E.T


def question_relevance(P2d, question, embed_tokens, tok):
    """(N,) cosine to the mean embedding of the question sentence (only used when the question has no noun)."""
    ids = tok(question, add_special_tokens=False, return_tensors='pt').input_ids[0]
    special = set(getattr(tok, 'all_special_ids', []) or [])
    if special:
        ids = ids[torch.tensor([int(i) not in special for i in ids], dtype=torch.bool)]
    with torch.no_grad():
        q = F.normalize(embed_tokens(ids.to(embed_tokens.weight.device)).float().mean(0), dim=-1)
    return F.normalize(P2d, dim=1) @ q.float().cpu()
