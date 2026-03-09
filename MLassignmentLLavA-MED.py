import argparse
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from PIL import Image
import json
import os
from tqdm import tqdm
from llava.mm_utils import get_model_name_from_path
def main(args):
    model_path = args.model_path
    conv_mode = args.conv_mode
    question_file = args.question_file
    image_folder = args.image_folder
    answers_file = args.answers_file
    temperature = args.temperature

    model_name = "llava_mistral"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name  # 最新版参数顺序是这样
    )

    conv = conv_templates[conv_mode].copy()

    with open(question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    answers = []
    for q in tqdm(questions):
        image_file = q['image']
        question = q['question']

        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].to(model.device, dtype=torch.float16)

        inp = f"{conv.system}\n<image>\nUSER: {question} ASSISTANT:"
        input_ids = tokenizer_image_token(inp, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(model.device),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=256,
            )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answers.append({"id": q['id'], "answer": output})

    with open(answers_file, 'w') as f:
        for ans in answers:
            f.write(json.dumps(ans) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--model-path", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
