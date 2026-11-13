# QNPruner

QNPruner is a training-free visual token pruner for LLaVA-3D. It keeps exactly **K** of the scene's voxel tokens, chosen by the nouns of the question, and runs before the LLM.

- **What it uses:** only the frozen LLaVA-3D parts (CLIP layer -2 features, the projector and the LLM input embeddings) and spaCy.
- **What it does not use:** training, an extra encoder, an LLM forward pass, ground truth, or any dataset- or K-specific setting.

The package ships code only. It contains no data, annotations, features or checkpoints.

## Method

| Step | File | What it does |
|---|---|---|
| 0 | `text.py` | spaCy parse of the question sentence: nouns, attribute (amod) phrases, and the question type. The type sets the Stage-1 share ρ: A = 20/32, B = 12/32, C (all other questions) = 1/2. |
| 1 | `relevance.py` | Relevance r_w(i) = cos(P(f2d_i), e(w)). P is the LLaVA-3D projector and e(w) is the mean LLM input embedding of w. |
| 2 | `stage1.py` | Stage 1 builds kb = round(ρK) anchors, described below. |
| 3 | `stage2.py` | Stage 2 fills the remaining kv = K − kb tokens, described below. |
| 4 | `merge.py` | NL2 merging, described below. |
| – | `pipeline.py` | Tower hook: builds the voxel features and calls the steps in order. |

**Stage 1 (kb anchors)**
1. Noun groups: assign each voxel to argmax_j r_j. Keep it only if its best relevance is at or above τ, the scene median.
2. Equal quotas: each noun gets an equal share of kn = round(kb/2), or of kb when the question has no attribute phrase.
3. Per-noun exploration: protect the ceil(q/2) most relevant voxels. Then take one voxel from each not-yet-covered 3D component (0.35 m single linkage, size ≥ 2), choosing components by peak relevance × appearance novelty.
4. Phrases: each attribute phrase gets an equal share of the rest. Its column is (1 − w)·z(r_head) + w·z(r_phrase), where w = spread_head / (spread_head + spread_phrase).
5. Relevance fill: any shortfall is filled from max_j r_j.

**Stage 2 (kv = K − kb tokens)**
1. Local FL: round(0.25·kv) picks by facility location on the appearance similarity. Candidates are valid voxels within 0.6 m of an anchor.
2. Global FL: the rest by facility location over all valid voxels.
   - Similarity: S = 0.5·(cos P(f_fused) + 1)/2 + 0.5·(cos f2d + 1)/2.
   - Coverage weight: a_u = 0.5/|U|.
   - Relevance bonus: 0.5·r̂/K, where r̂ = (max_j r_j + 1)/2.

**No-noun questions:** max-min diversity selects all K tokens.

**NL2 (Step 4)**
1. Each unselected valid voxel is assigned to its most similar selected token, using the pre-projector fused features.
2. It is merged only if that token is a Stage-1 anchor within 2 hops on the 26-neighbour voxel graph.
3. The anchor row becomes the uniform mean of itself and its merged voxels. Token IDs and K do not change.

The LLM receives the K rows in voxel order. Every greedy tie goes to the lower voxel index.

## Requirements

**Code**
- This repository's `llava/` package (LLaVA-3D with the external-method hook `llava/model/multimodal_encoder/method_registry.py`) and `scripts/eval/`.
- Python packages: torch 2.1.2 (CUDA 11.8 build), transformers 4.37.2, torch_scatter, scipy, spaCy 3.8.16 with `en_core_web_sm` 3.8.0, plus pycocoevalcap and mmengine for the evaluators.
- Exact versions are listed in `docker-pip-constraints.txt` and `docker-extra-requirements.txt`.

**Models** (downloaded from HuggingFace on first use)
- `ChaimZhu/LLaVA-3D-7B`
- `openai/clip-vit-large-patch14-336` (the vision tower)

**Data** (not included; prepare it as in the LLaVA-3D instructions)

| Path | Contents |
|---|---|
| `/data/LLaVA-3D-Pretrain/scannet/<scene>/` | ScanNet RGB-D frames, poses and intrinsics. Set the root with `--video-folder`. |
| `/data/LLaVA-3D-Pretrain/openeqa_ep600` | OpenEQA episode frames (`OPENEQA_VIDEO_FOLDER`). |
| `playground/data/annotations/llava-3d-scanqa_val_question.json`, `llava3d_scanqa_val_answer.json` | ScanQA val questions / answers (LLaVA-3D format) |
| `playground/data/annotations/llava-3d-sqa3d_test_question.json`, `llava3d_sqa3d_test_answer.json` | SQA3D test questions / answers |
| `playground/data/annotations/llava-3d-openeqa_scannet.json`, `open-eqa-v0.json` | OpenEQA, ScanNet subset |
| `playground/data/annotations/embodiedscan_infos_full.json` | Scene metadata read by the LLaVA-3D loaders |

## Usage

Run from the repository root. The second argument is the voxel size (0.2 m) and the third is K.

```bash
bash scripts/eval/eval_scanqa.sh  qnpruner 0.2 64     # ScanQA val: EM, CIDEr, ...
bash scripts/eval/eval_sqa3d.sh   qnpruner 0.2 64     # SQA3D test: EM
OPENEQA_VIDEO_FOLDER=/data/LLaVA-3D-Pretrain/openeqa_ep600 \
bash scripts/eval/eval_openeqa.sh qnpruner 0.2 64     # OpenEQA predictions
python scripts/eval/score_openeqa.py <pred.json> --model gpt-4o   # LLM-Match
```

`score_openeqa.py` prompts for the OpenAI key at run time. The key is never stored.

**Optional settings**
- `OURS_STATS_FILE=<file.jsonl>` writes one record per question: `selected_indices`, `cluster_indices` (Stage-1 anchors), `qa_order`, nouns, phrases, question type and timings.
- `DETERMINISTIC=1 PYTHONPATH=$PWD/determinism` enables deterministic CUDA kernels and per-question seeds. With the same GPU type and software versions this reproduces our numbers exactly; other setups may differ slightly.
- `OURS_TOKENIZER` selects the tokenizer, which defaults to the checkpoint's own.

Batch size must be 1, as in the LLaVA-3D evaluation scripts.
