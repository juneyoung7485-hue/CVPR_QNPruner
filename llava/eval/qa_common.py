"""Shared loop of the ScanQA / SQA3D / OpenEQA evaluators: load LLaVA-3D, read each scene's RGB-D frames, generate.

Greedy decoding (temperature 0), max_new_tokens 512, batch size 1, as in the LLaVA-3D evaluation scripts. The raw
question (SQA3D: situation + question) is passed to the video tower as ``query_texts``; only QNPruner reads it.
"""
import argparse
import json
import math
import os

import shortuuid
import torch
from tqdm import tqdm

from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_videos
from llava.model.builder import load_pretrained_model
from llava.sample_context import SAMPLE
from llava.utils import disable_torch_init


def get_chunk(lst, n, k):
    """The k-th of n (roughly) equal chunks of lst."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)][k]


def base_parser(question_file, answers_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="ChaimZhu/LLaVA-3D-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default=question_file)
    parser.add_argument("--answers-file", type=str, default=answers_file)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions (0 = all).")
    return parser


def load_questions(args):
    with open(args.question_file, 'r') as fh:
        questions = json.load(fh)
    if not questions:
        raise ValueError('The question file is empty.')
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.limit > 0:
        questions = questions[:args.limit]
    return questions


def load_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    return tokenizer, model, processor, model_name


def chat_prompt(qs, conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def generate(args, tokenizer, model, processor, input_ids, video_path, video_file, question_id, cur_prompt,
             clicks=None):
    """The decoded answer for one question."""
    videos_dict = process_videos(video_path, processor['video'], mode='uniform', device=model.device, text=cur_prompt)
    kwargs = dict(
        images=videos_dict['images'].to(model.device, dtype=torch.bfloat16),
        depths=videos_dict['depths'].to(model.device, dtype=torch.bfloat16),
        poses=videos_dict['poses'].to(model.device, dtype=torch.bfloat16),
        intrinsics=videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16),
    )
    if clicks is not None:
        kwargs['clicks'] = clicks
    SAMPLE['scene_id'] = video_file
    SAMPLE['question_id'] = question_id
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            image_sizes=None,
            query_texts=[cur_prompt],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            use_cache=True,
            **kwargs,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def open_answers(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or '.', exist_ok=True)
    return open(answers_file, "w")


def write_answer(fh, record, model_name):
    record.update(answer_id=shortuuid.uuid(), model_id=model_name, metadata={})
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def progress(questions):
    return tqdm(questions)
