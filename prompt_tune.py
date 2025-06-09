"""
Prompt Tuning code for llava-med
"""


import json
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import PromptTuningConfig, get_peft_model, TaskType
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

# === CONFIG ===
json_path = "train_data.json"  
model_name = "microsoft/llava-med-v1.5-mistral-7b"
output_dir = "./llava_prompt_tuned_model"
num_virtual_tokens = 20
max_input_len = 640
max_output_len = 128
batch_size = 1
num_epochs = 2
learning_rate = 5e-4

# === LOAD & FORMAT DATA ===
def load_llava_json(json_path):
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    formatted = []
    for example in raw_data:
        conv = example["conversations"]
        image_path = example["image"]
        question = conv[0]["value"].replace("<image>", "").strip()
        answer = conv[1]["value"].strip()

        formatted.append({
            "image": image_path,
            "input": f"### Human: <image>\n{question}\n### Assistant:",
            "output": answer
        })

    return Dataset.from_list(formatted)

dataset = load_llava_json(json_path)

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = LlavaMistralForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )
model.gradient_checkpointing_enable()
# === SETUP PROMPT TUNING ===
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="### Human: <image>\nDoes this image show a",
    num_virtual_tokens=num_virtual_tokens,
    tokenizer_name_or_path=model_name
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === TOKENIZE DATA ===
def tokenize_example(example):
    full_prompt = example["input"] + " " + example["output"]
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        padding="max_length",
        max_length=max_input_len
    )
    
    input_ids = tokenized["input_ids"]
    labels = input_ids.copy()

    input_len = len(tokenizer(example["input"], truncation=True, max_length=max_input_len)["input_ids"])
    labels[:input_len] = [-100] * input_len

    tokenized["labels"] = labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)

# === TRAINING SETUP ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True, 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# === TRAIN ===
trainer.train()

# === SAVE ===
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Prompt-tuned model saved to {output_dir}")
