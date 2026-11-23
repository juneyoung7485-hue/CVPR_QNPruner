"""ScanQA val predictions with LLaVA-3D (POOLING=voxelize) or LLaVA-3D + QNPruner (POOLING=qnpruner TARGET_TOKENS=K)."""
import os

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.eval import qa_common as Q
from llava.mm_utils import tokenizer_image_token


def eval_model(args):
    questions = Q.load_questions(args)
    tokenizer, model, processor, model_name = Q.load_model(args)
    ans_file = Q.open_answers(args)
    for line in Q.progress(questions):
        idx = line["question_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        prompt = Q.chat_prompt(qs, args.conv_mode)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        outputs = Q.generate(args, tokenizer, model, processor, input_ids, video_path, video_file, idx, cur_prompt)
        Q.write_answer(ans_file, {"question_id": idx, "prompt": cur_prompt, "text": outputs}, model_name)
    ans_file.close()


if __name__ == "__main__":
    parser = Q.base_parser("playground/data/annotations/llava-3d-scanqa_val_question.json",
                           "./pred/scanqa_answers.json")
    eval_model(parser.parse_args())
