"""OpenEQA LLM-Match scoring (official open-eqa evaluate-predictions.py logic) with a key typed in at run time.

The key is read with getpass and handed to the openai client in this process only: it is never written to a file,
an environment variable or a log, and any OPENAI_API_KEY inherited from the (shared) server environment is ignored.
Scores are resumable: an existing <stem>-metrics.json is extended, not recomputed.

Usage (needs a terminal for the key prompt; run with the open-eqa venv):
  ~/VILAB/jyoung/open-eqa/.venv/bin/python scripts/eval/score_openeqa.py pred/openeqa_*_official.json \
      [--model gpt-4-1106-preview] [--workers 8]
"""
import argparse
import getpass
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OPEN_EQA = Path(os.path.expanduser('~/VILAB/jyoung/open-eqa'))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OPEN_EQA))
os.environ.pop('OPENAI_API_KEY', None)          # never fall back to someone else's key

import re  # noqa: E402
import openai  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_random_exponential  # noqa: E402
from openeqa.evaluation.llm_match import parse_score  # noqa: E402
from openeqa.utils.openai_utils import prepare_openai_messages  # noqa: E402
from openeqa.utils.prompt_utils import load_prompt  # noqa: E402


def llm_match(client, model, question, answer, prediction, extra_answers=None):
    """openeqa.evaluation.llm_match.get_llm_match_score with an explicit client: upstream's call_openai_api builds
    openai.OpenAI() from the environment, which would pick up the shared server key. Same prompt, seed 1234,
    max_tokens 32, temperature 0.2 and retry policy as upstream."""
    if prediction is None:
        return 0
    prompt = load_prompt('mmbench' if extra_answers is None else 'mmbench-extra')
    messages = prepare_openai_messages(prompt.format(question=question, answer=answer, prediction=prediction,
                                                     extra_answers=extra_answers))
    out = _call(client, model, messages)
    try:
        return parse_score(out)
    except ValueError:
        # 2026-11-10: judge sometimes answers "2\n<explanation>" (no "Your mark:" tag, not pure digits) -> upstream raises
        # and kills the scoring run. Take a leading 1-5 mark in that case only; anything else still raises.
        m = re.match(r'\s*([1-5])\b', out or '')
        if m:
            return int(m.group(1))
        raise


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def _call(client, model, messages):
    c = client.chat.completions.create(model=model, messages=messages, seed=1234, max_tokens=32, temperature=0.2)
    assert len(c.choices) == 1
    return c.choices[0].message.content


def preprocess(answer):
    # as evaluate-predictions.py: drop anything after the last period
    if answer:
        end = answer.rfind('.')
        if end >= 0 and end + 1 < len(answer):
            answer = answer[:end + 1]
    return answer


def score_file(path, items, client, model, workers, out_dir):
    results = json.load(open(path))
    assert {r['question_id'] for r in results} == set(items) and len(results) == len(items), \
        f'{path}: predictions do not cover the {len(items)} ScanNet questions'
    out = out_dir / (Path(path).stem + '-metrics.json')
    scores = json.load(open(out)) if out.exists() else {}
    todo = [r for r in results if r['question_id'] not in scores]

    def one(r):
        it = items[r['question_id']]
        return r['question_id'], llm_match(client, model, it['question'], it['answer'],
                                           preprocess(r['answer']), it.get('extra_answers'))

    with ThreadPoolExecutor(workers) as ex:
        for i, (qid, s) in enumerate(ex.map(one, todo), 1):
            scores[qid] = s
            if i % 50 == 0 or i == len(todo):
                json.dump(scores, open(out, 'w'), indent=2)
                print(f'  {Path(path).name}: {len(scores)}/{len(items)}', flush=True)
    s = np.array(list(scores.values()), dtype=float)
    return float(np.mean(100.0 * (np.clip(s, 1, 5) - 1) / 4)), len(scores), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--model', default='gpt-4-1106-preview', help='official OpenEQA judge')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-dir', default=str(ROOT / 'results/repro_h100_gpu6/openeqa_metrics'))
    args = ap.parse_args()

    data = json.load(open(OPEN_EQA / 'data/open-eqa-v0.json'))
    items = {d['question_id']: d for d in data if d['episode_history'].startswith('scannet-v0/')}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = getpass.getpass('OpenAI API key (not echoed, not saved): ').strip()
    # check the key and the judge model once before scoring anything
    client = openai.OpenAI(api_key=key)
    del key
    try:
        client.models.retrieve(args.model)
    except Exception as e:
        sys.exit(f'key/model check failed for {args.model}: {type(e).__name__}: {e}')

    summary = out_dir / 'summary.csv'
    for p in args.results:
        score, n, out = score_file(p, items, client, args.model, args.workers, out_dir)
        print(f'{Path(p).name}: LLM-Match {score:.2f} (n={n}, judge={args.model})', flush=True)
        new = not summary.exists()
        with open(summary, 'a') as fh:
            if new:
                fh.write('results,n,judge,LLM_Match,metrics\n')
            fh.write(f'{p},{n},{args.model},{score:.4f},{out}\n')


if __name__ == '__main__':
    main()
