"""
This is our evaluation code for prompt tuning

"""

import argparse
import os
import pandas as pd
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# === Eval logic: classify response ===
def classify_response(response):
    response = response.lower()
    if any(kw in response for kw in ["no wilms", "no evidence", "not present", "absent", "no", "not show"]):
        return 0
    elif any(kw in response for kw in ["yes", "sign of wilms", "shows wilms", "shows a wilms"]):
        return 1
    else:
        return 1

# === Main function ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="./Wilms_Tumor/_classes.csv")
    parser.add_argument('--image_root', type=str, default="./Wilms_Tumor/test/")
    parser.add_argument('--adapter_path', type=str, default="./llava_prompt_tuned_model")
    parser.add_argument('--base_model_path', type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument('--output_csv', type=str, default="prompt_output.csv")
    parser.add_argument('--question', type=str, default="Does the image show a Wilms tumor?")
    args = parser.parse_args()

    # === Load model ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.base_model_path,
        model_base=None,
        model_name=args.base_model_path,
        load_8bit=False,
        load_4bit=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    llava_model = model.base_model
    llava_model.eval()

    # === Load CSV ===
    df = pd.read_csv(args.csv_path)
    df.columns = df.columns.str.strip()

    results = []
    tp = tn = fp = fn = total = correct = 0

    for idx, row in df.iterrows():
        filename = row['filename']
        gt = int(row['wilms_tumor'])
        image_path = os.path.join(args.image_root, filename)

        if not os.path.exists(image_path):
            print(f"❌ Not found: {filename}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

            prompt = f"{DEFAULT_IMAGE_TOKEN}\nUSER: {args.question}\nASSISTANT:"
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            image_sizes = torch.tensor([image_tensor.shape[-2:]], device=model.device)

            with torch.no_grad():
                output = model.base_model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0.2,
                    max_new_tokens=100,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
            pred = classify_response(output_text)

            result = {
                'filename': filename,
                'ground_truth': gt,
                'model_response': output_text,
                'predicted': pred,
                'correct': pred == gt
            }
            results.append(result)

            if pred == gt:
                correct += 1
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 1:
                fn += 1

            total += 1
            print(f"[{idx+1}] {filename}: GT={gt}, PRED={pred}, ✅={pred==gt}, Text='{output_text[:80]}'")

        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
            continue

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_csv, index=False)
    acc = correct / total if total else 0

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.4f}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    if tp + fp:
        print(f"Precision: {tp / (tp + fp):.4f}")
    if tp + fn:
        print(f"Recall: {tp / (tp + fn):.4f}")

if __name__ == '__main__':
    main()
