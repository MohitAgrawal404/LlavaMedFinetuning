"""
This is how you merge Lora's delta with LLava-med
"""

from peft import PeftModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
model_path = "./merged_llava_med_model_epoch_300"
# model_path = "microsoft/llava-med-v1.5-mistral-7b"
lora_path = "./checkpoints/llava-med-lora-alpha-16-64/checkpoint-400"

print("Loading base model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=False,
    load_4bit=False,
    device_map=None  # Don't use auto device mapping
)

# Move to GPU manually if you have one
if torch.cuda.is_available():
    model = model.cuda()

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained("./merged_llava_med_model_mistral-64")
tokenizer.save_pretrained("./merged_llava_med_model_mistral-64")

print("Merge complete!")
