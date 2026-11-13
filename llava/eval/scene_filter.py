"""Resolve a ``--scene`` value against a benchmark question file."""
from pathlib import Path


def scene_ids(scene):
    """Normalize a --scene value into the requested scene IDs.

    Accepts one scene, a comma-separated list, or an already-split sequence,
    each entry being a bare ID (``scene0011_00``) or any path ending in one
    (``scannet/scene0011_00``, ``/data/.../scene0011_00/``). An empty value
    means "every scene in the question file" and yields ``[]``.
    """
    if not scene:
        return []
    items = scene.split(',') if isinstance(scene, str) else list(scene)
    ids = []
    for item in items:
        name = Path(str(item).strip().rstrip('/')).name
        if name and name not in ids:
            ids.append(name)
    return ids


def filter_scene_questions(questions, scene):
    """Select exact ScanNet scene IDs, preserving annotation order."""
    wanted = scene_ids(scene)
    if not wanted:
        return questions
    wanted_set = set(wanted)
    selected = [q for q in questions if Path(q['video'].rstrip('/')).name in wanted_set]
    # Report every unmatched scene at once: a 20-scene sweep should not have to
    # be restarted once per typo.
    missing = [name for name in wanted
               if not any(Path(q['video'].rstrip('/')).name == name for q in selected)]
    if missing:
        raise ValueError(f'No questions for scene(s) {missing} in the question file.')
    return selected
