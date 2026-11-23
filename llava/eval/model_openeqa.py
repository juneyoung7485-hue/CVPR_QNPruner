"""OpenEQA (ScanNet) predictions with LLaVA-3D (POOLING=voxelize) or LLaVA-3D + QNPruner (POOLING=qnpruner
TARGET_TOKENS=K). Record and prompt format as LLaVA-3D's official llava/eval/model_openeqa.py."""
import os

import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from llava.eval import qa_common as Q
from llava.mm_utils import tokenizer_special_token


def eval_model(args):
    questions = Q.load_questions(args)
    tokenizer, model, processor, model_name = Q.load_model(args)
    ans_file = Q.open_answers(args)
    for line in Q.progress(questions):
        idx = line["prompt_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line['conversations'][0]['value']
        # cur_prompt = the plain question (what QNPruner parses); the annotation keeps "<video>\n" in front of it
        cur_prompt = qs[len(DEFAULT_VIDEO_TOKEN + '\n'):] if qs.startswith(DEFAULT_VIDEO_TOKEN + '\n') \
            else qs.replace(DEFAULT_VIDEO_TOKEN, '').strip()
        qs = qs.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
        if 'target' in line:
            boxes = line['target']['boxes']
            clicks = torch.tensor([[round(coord, 3) for coord in box[:3]] for box in boxes])
            qs = qs.replace('<boxes>', ('<boxes>, ' * len(boxes)).rstrip(', '))
        else:
            clicks = torch.zeros((0, 3))
        prompt = Q.chat_prompt(qs, args.conv_mode)
        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        outputs = Q.generate(args, tokenizer, model, processor, input_ids, video_path, video_file, idx, cur_prompt,
                             clicks=clicks.to(model.device, dtype=torch.bfloat16))
        Q.write_answer(ans_file, {"question_id": idx, "prompt": cur_prompt, "text": outputs, "video": video_file,
                                  "category": line.get("category")}, model_name)
    ans_file.close()


if __name__ == "__main__":
    parser = Q.base_parser("playground/data/annotations/llava-3d-openeqa_scannet.json",
                           "./pred/openeqa_answers.json")
    eval_model(parser.parse_args())
