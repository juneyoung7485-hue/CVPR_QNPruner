"""The question being evaluated, published by the eval loops (llava/eval/model_*.py) before each generate call.

QNPruner reads it only to label its optional per-question record (OURS_STATS_FILE); the selection never uses it.
"""
SAMPLE = {}
