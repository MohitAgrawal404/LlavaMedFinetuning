# LLaVA Fine-tuning Project

This project is forked from [LLaVA](https://github.com/haotian-liu/LLaVA) and provides implementations for fine-tuning multimodal models using two different approaches: LoRA (Low-Rank Adaptation) and Prompt Tuning.

## Overview

The repository supports fine-tuning LLaVA models for specialized tasks, particularly focusing on medical image analysis. Two fine-tuning strategies are available:

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using Low-Rank Adaptation
- **Prompt Tuning**: Fine-tuning through learnable prompt embeddings

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended)
- Required dependencies from the original LLaVA project

## Installation

1. Clone this repository:
```bash
git clone git@github.com:MohitAgrawal404/llava-med-finetuning.git
cd llava-med-finetuning/llava
```

2. Install dependencies:
```bash
conda env create
```

## Fine-tuning Methods

### Method 1: LoRA Fine-tuning

LoRA (Low-Rank Adaptation) provides an efficient way to fine-tune large models by learning low-rank decomposition matrices.

#### Step 1: Run LoRA Fine-tuning

Execute the LLaVA-Med fine-tuning script:

```bash
bash llava-med.sh
```

This script utilizes LLaVA's built-in LoRA fine-tuning capabilities to train the model on your dataset.

#### Step 2: Merge LoRA Weights

After fine-tuning completes, merge the LoRA weights with the base LLaVA-Med model:

```bash
python combine_lora.py
```

This will create a merged model that incorporates the learned LoRA adaptations.

#### Step 3: Evaluate LoRA Model

Evaluate the fine-tuned LoRA model using:

```bash
python eval_lora.py \
    --output_path "./output.csv" \
    --image_folder "./Wilms_Tumor/test/" \
    --csv_path "./Wilms_Tumor/_classes.csv" \
    --model_path "./merged_llava_med_model_mistral-64"
```

**Parameters:**
- `--output_path`: Path where evaluation results will be saved
- `--image_folder`: Directory containing test images
- `--csv_path`: Path to CSV file with class labels
- `--model_path`: Path to the merged model directory

### Method 2: Prompt Tuning

Prompt tuning fine-tunes the model by learning task-specific prompt embeddings while keeping the base model parameters frozen.

#### Step 1: Run Prompt Tuning

Execute the prompt tuning script:

```bash
python prompt_tune.py
```

This will fine-tune the model using learnable prompt embeddings.

#### Step 2: Evaluate Prompt-Tuned Model

Evaluate the prompt-tuned model using:

```bash
python eval_prompt.py \
    --output_path "./output.csv" \
    --image_folder "./Wilms_Tumor/test/" \
    --csv_path "./Wilms_Tumor/_classes.csv" \
    --model_path "./llava_prompt_tuned_model"
```

**Parameters:**
- `--output_path`: Path where evaluation results will be saved
- `--image_folder`: Directory containing test images
- `--csv_path`: Path to CSV file with class labels
- `--model_path`: Path to the prompt-tuned model directory


## Troubleshooting

- Ensure CUDA is available if using GPU acceleration
- Verify that image paths in your CSV file match the actual image locations
- Check that model paths exist and contain the necessary model files
- Monitor GPU memory usage during training to avoid out-of-memory errors

## Contributing

This project builds upon the excellent work of the LLaVA team. Please refer to the original LLaVA repository for additional documentation and community guidelines.

## License

This project inherits the license from the original LLaVA project. Please refer to the LICENSE file for details.

## Citation

If you use this code in your research, please cite both this work and the original LLaVA paper:

```bibtex
@misc{liu2023llava,
    title={Visual Instruction Tuning}, 
    author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
    year={2023},
    eprint={2304.08485},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```