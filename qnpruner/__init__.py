"""QNPruner: query-noun guided, training-free visual token pruning for LLaVA-3D (exactly K voxel tokens).

Use it as a LLaVA-3D method package: ``POOLING=qnpruner`` (see README.md). Pipeline (one module per step):
    text.py       Step 0  spaCy parse: question nouns, amod phrases, question type -> Stage-1 share rho
    relevance.py  Step 1  r_w(i) = cos(P(f2d_i), e(w)) with the frozen projector P and LLM input embeddings e
    stage1.py     Step 2  noun groups, equal quotas, 3D-component exploration, phrase columns, relevance fill
    stage2.py     Step 3  local appearance FL + global coverage FL with a relevance bonus; no-noun max-min path
    merge.py      Step 4  NL2: merge nearby unselected voxels into the Stage-1 anchors (uniform mean)
    pipeline.py   tower hook: voxel features, selection, merged rows, optional per-question log (OURS_STATS_FILE)
No training, no extra encoder, no LLM forward, no ground truth, no dataset- or K-specific branch.
"""
from .pipeline import NAME, compress, select  # noqa: F401

QUERY_DEPENDENT = True      # the output depends on the question: LLaVA-3D's per-scene feature cache must not serve it
