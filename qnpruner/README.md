# QNPruner

QNPruner is a training-free visual token pruner for LLaVA-3D. It keeps exactly **K** of the scene's voxel tokens, chosen by the nouns of the question, and runs before the LLM.

- **What it uses:** only the frozen LLaVA-3D parts (CLIP layer -2 features, the projector and the LLM input embeddings) and spaCy.
- **What it does not use:** training, an extra encoder, an LLM forward pass, ground truth, or any dataset- or K-specific setting.

The package ships code only. It contains no data, annotations, features or checkpoints.

## Method

| Step | File | What it does |
|---|---|---|
| 0 | `text.py` | spaCy parse of the question sentence (the whole prompt when no single final question sentence can be isolated): nouns, attribute (amod) phrases, and the question type. The type sets the Stage-1 share ρ: A = 20/32, B = 12/32, C (all other questions, including those without a noun) = 1/2. |
| 1 | `relevance.py` | Relevance r_w(i) = cos(P(f2d_i), e(w)). P is the LLaVA-3D projector and e(w) is the mean LLM input embedding of w. |
| 2 | `stage1.py` | Stage 1 builds kb = round(ρK) anchors, described below. |
| 3 | `stage2.py` | Stage 2 fills the remaining kv = K − kb tokens, described below. |
| 4 | `merge.py` | NL2 merging, described below. |
| – | `pipeline.py` | Tower hook: builds the voxel features and calls the steps in order. |

**Stage 1 (kb anchors)**
1. Noun groups: assign each voxel to argmax_j r_j. Keep it only if its best relevance is at or above τ, the scene median.
2. Equal quotas: each noun gets an equal share of kn = round(kb/2), or of kb when the question has no attribute phrase.
   - Noun order: a noun counts when it occurs in the question sentence, but the nouns are ordered by their first mention in the full prompt (situation text included); the remainder of the equal split goes to the earlier nouns.
3. Per-noun exploration: protect the ceil(q/2) most relevant voxels. Then take one voxel from each not-yet-covered 3D component (0.35 m single linkage, size ≥ 2), choosing components by [peak relevance]₊ × appearance novelty, where [x]₊ = max(x, 0). If every candidate scores 0, the component that contains the lowest voxel index is taken.
4. Phrases: each attribute phrase gets an equal share of the rest. Its column is (1 − w)·z(r_head) + w·z(r_phrase), where w = spread_head / (spread_head + spread_phrase).
5. Relevance fill: any shortfall is filled from max_j r_j.

**Stage 2 (kv = K − kb tokens)**
1. Local FL: round(0.25·kv) picks by facility location on the appearance similarity. Candidates are valid voxels within 0.6 m of an anchor.
2. Global FL: the rest by facility location over all valid voxels.
   - Similarity: S = 0.5·(cos P(f_fused) + 1)/2 + 0.5·(cos f2d + 1)/2.
   - Coverage weight: a_u = 0.5/|U|.
   - Relevance bonus: 0.5·r̂/K, where r̂ = (max_j r_j + 1)/2.

**No-noun questions:** query-agnostic max-min diversity on the projected features selects all K tokens (Stage 2 is skipped), and NL2 merges into these tokens.

**NL2 (Step 4)**
1. Each unselected valid voxel is assigned to its most similar selected token, using the pre-projector fused features.
2. It is merged only if that token is a Stage-1 anchor within 2 hops on the 26-neighbour voxel graph.
3. The anchor row becomes the uniform mean of itself and its merged voxels. Token IDs and K do not change.

The LLM receives the K rows in voxel order. Every greedy tie goes to the lower voxel index.

## Use

Select it in LLaVA-3D's video tower with `POOLING=qnpruner TARGET_TOKENS=<K>` (`VOXEL_SIZE=0.2`), e.g. through
`scripts/eval/eval_scanqa.sh qnpruner 0.2 64`. Installation, data and evaluation are described in the repository
[README](../README.md). Batch size 1.
