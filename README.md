# QNPruner

Training-free, query-aware visual token pruning for 3D vision-language models (LLaVA-3D).

QNPruner keeps exactly **K** of the scene's voxel tokens, chosen by the nouns of the question, before the LLM runs. It uses only frozen LLaVA-3D parts (CLIP layer -2 features, the projector and the LLM input embeddings) and spaCy: no training, no extra encoder, no LLM forward pass, no ground truth and no dataset- or K-specific setting. The method is described step by step in [`qnpruner/README.md`](qnpruner/README.md).

This repository contains code only. It contains no data, annotations, features or model weights.

## Repository layout

| Path | Contents |
|---|---|
| `qnpruner/` | QNPruner (one module per step). Selected with `POOLING=qnpruner`. |
| `llava/` | LLaVA-3D model and evaluation code. Modified only to hand the selector what it needs: the external-method hook (`llava/model/multimodal_encoder/method_registry.py`, `video_encoder.py`), the question text and the LLM embedding table (`llava/model/llava_arch.py`), and compact ScanQA / SQA3D / OpenEQA evaluators (`llava/eval/`). |
| `scripts/eval/` | `eval_scanqa.sh`, `eval_sqa3d.sh`, `eval_openeqa.sh`, and `score_openeqa.py` (OpenEQA LLM-Match judge). |
| `determinism/` | `sitecustomize.py` for deterministic kernels and per-question seeds (used via `PYTHONPATH`). |

## Installation

Our experiments ran in a Docker image; its pinned versions are in `docker-pip-constraints.txt` and `docker-extra-requirements.txt`. A typical setup (not tested from scratch):

```bash
pip install -e .                                   # LLaVA-3D dependencies (see pyproject.toml)
pip install -c docker-pip-constraints.txt -r docker-extra-requirements.txt
pip install torch-scatter spacy==3.8.16 pycocoevalcap mmengine openai
python -m spacy download en_core_web_sm            # en_core_web_sm 3.8.0
```

Tested with torch 2.1.2 (CUDA 11.8) and transformers 4.37.2 on one H100.

**Models** (downloaded from HuggingFace on first use): `ChaimZhu/LLaVA-3D-7B` and `openai/clip-vit-large-patch14-336` (the vision tower).

**Data** (not included; prepare it as in the LLaVA-3D instructions)

| Path | Contents |
|---|---|
| `/data/LLaVA-3D-Pretrain/scannet/<scene>/` | ScanNet RGB-D frames, poses and intrinsics (`--video-folder` of the evaluators). |
| `/data/LLaVA-3D-Pretrain/openeqa_ep600` | OpenEQA episode frames (`OPENEQA_VIDEO_FOLDER`). |
| `playground/data/annotations/llava-3d-scanqa_val_question.json`, `llava3d_scanqa_val_answer.json` | ScanQA val questions / answers (LLaVA-3D format) |
| `playground/data/annotations/llava-3d-sqa3d_test_question.json`, `llava3d_sqa3d_test_answer.json` | SQA3D test questions / answers |
| `playground/data/annotations/llava-3d-openeqa_scannet.json` | OpenEQA questions, ScanNet part (LLaVA-3D format) |
| `playground/data/annotations/embodiedscan_infos_full.json` | Scene metadata read by the LLaVA-3D loaders |

LLM-Match scoring also needs a checkout of [open-eqa](https://github.com/facebookresearch/open-eqa) (its `data/open-eqa-v0.json` and prompts), given by `OPEN_EQA_ROOT` (default `./open-eqa`).

## Usage

Run from the repository root. Arguments: method (`qnpruner`, or `native` for LLaVA-3D without token selection), voxel size (0.2 m in all our experiments) and K.

```bash
export DETERMINISTIC=1 PYTHONPATH=$PWD/determinism   # optional: deterministic kernels (see below)

bash scripts/eval/eval_scanqa.sh  qnpruner 0.2 64      # ScanQA val: EM, BLEU, METEOR, ROUGE, CIDEr
bash scripts/eval/eval_sqa3d.sh   qnpruner 0.2 64      # SQA3D test: EM
OPENEQA_VIDEO_FOLDER=/data/LLaVA-3D-Pretrain/openeqa_ep600 \
bash scripts/eval/eval_openeqa.sh qnpruner 0.2 64      # OpenEQA predictions (+ *_official.json)
python scripts/eval/score_openeqa.py pred/openeqa_qnpruner_vs0.2_t64_official.json \
       --model gpt-4o-2024-08-06                       # LLM-Match (judge used in the paper)

bash scripts/eval/eval_scanqa.sh native                # LLaVA-3D, all voxel tokens
```

Predictions are written to `pred/` (one JSON line per question). `score_openeqa.py` asks for the OpenAI key at run time and never stores it.

**Options**
- `EVAL_LIMIT=N` evaluates only the first N questions (the prediction file gets a `_subset<N>` suffix).
- `OURS_STATS_FILE=<file.jsonl>` writes one record per question: `selected_indices` (the K kept voxel ids), `cluster_indices` (Stage-1 anchors), nouns, phrases, question type, K and the number of valid voxels.
- `DETERMINISTIC=1 PYTHONPATH=$PWD/determinism` enables deterministic CUDA kernels and per-question seeds. With the same GPU type and software versions this reproduces our numbers exactly; other setups may differ slightly.
- `OURS_TOKENIZER` selects the tokenizer used for the word embeddings (default: the checkpoint's own).

Batch size is 1, as in the LLaVA-3D evaluation scripts.

## License

- `qnpruner/` is our code.
- `llava/` and the evaluation scripts are modified from [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D), which is released under CC BY-NC-SA 4.0. They are redistributed under the same license.

See `NOTICE` for details.

## Citation

Coming soon.
