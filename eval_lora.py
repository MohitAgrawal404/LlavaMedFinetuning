"""
This is our evaluation code for Lora finetuning after you've merged it with llava-med

"""

import pandas as pd
import torch
from PIL import Image
import os
import argparse
from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
import warnings
import re
from peft import PeftModel
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned LLaVA model and tokenizer"""
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map="auto"
    )
    return tokenizer, model, image_processor, context_len

def prepare_conversation(question, conv_mode="v1"):
    """Prepare the conversation template"""
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt

def evaluate_image(model, tokenizer, image_processor, image_path, question, device):
    """Evaluate a single image with the given question"""
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # Process image with proper configuration
        if hasattr(model.config, 'image_aspect_ratio'):
            aspect_ratio = model.config.image_aspect_ratio
        else:
            aspect_ratio = 'pad'
            
        image_tensor = process_images([image], image_processor, model.config)
        
        if image_tensor.dim() == 5:  # Remove extra batch dimension if present
            image_tensor = image_tensor.squeeze(0)
        
        image_tensor = image_tensor.to(device, dtype=torch.float16)
        
        # Prepare prompt
        prompt = prepare_conversation(question)
        
        # Tokenize with proper handling
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # Ensure proper dimensions
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        input_ids = input_ids.to(device)
        
        # Generate response with error handling
        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=128,  # Reduced for stability
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=None
                )
            except RuntimeError as e:
                if "size mismatch" in str(e).lower():
                    print(f"  Tensor size mismatch, trying alternative approach...")
                    # Try with different image processing
                    image_tensor = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=64,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    raise e
        
        # Decode response
        outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
        return outputs
    
    except Exception as e:
        print(f"  Error processing {image_path}: {str(e)}")
        return "Error"

def classify_response(response):
    """Classify response as yes/no based on presence of 'yes' in the answer"""
    if response == "Error":
        return -1  # Special case for errors
    if "yes" in response.lower() or "shows a wilms tumor" in response.lower():
        return 1
    else:
        return 0

def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned LLaVA model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the CSV file with evaluation data')
    parser.add_argument('--image_folder', type=str, default='./',
                       help='Folder containing the images')
    parser.add_argument('--output_path', type=str, default='evaluation_results.csv',
                       help='Path to save evaluation results')
    parser.add_argument('--question', type=str, default="Does the image show a Wilms tumor?",
                       help='Question to ask for each image')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer, model, image_processor, context_len = load_model_and_tokenizer(args.model_path)
    print("Model loaded successfully!")
    
    # Load CSV data
    df = pd.read_csv(args.csv_path)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Loaded {len(df)} samples from CSV")
    print(f"Available columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_cols = ['filename', 'wilms_tumor']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Initialize results
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    print("\nStarting evaluation...")
    print("-" * 50)
    
    for idx, row in df.iterrows():
        filename = row['filename']
        ground_truth = row['wilms_tumor']  # 1 if Wilms tumor, 0 if not
        
        image_path = os.path.join(args.image_folder, filename)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        print(f"Processing {idx+1}/{len(df)}: {filename}")
        
        # Get model response
        response = evaluate_image(model, tokenizer, image_processor, image_path, args.question, device)
        
        # Classify response
        predicted = classify_response(response)
        
        # Handle errors separately
        if predicted == -1:
            print(f"  Skipping due to error")
            continue
        
        # Check if correct
        is_correct = (predicted == ground_truth)
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Store results
        result = {
            'filename': filename,
            'ground_truth': ground_truth,
            'model_response': response,
            'predicted': predicted,
            'correct': is_correct
        }
        results.append(result)
        
        print(f"  Ground truth: {ground_truth}, Predicted: {predicted}, Correct: {is_correct}")
        print(f"  Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        print()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    
    # Print summary
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total samples: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Results saved to: {args.output_path}")
    
    # Print confusion matrix info
    tp = sum(1 for r in results if r['ground_truth'] == 1 and r['predicted'] == 1)
    tn = sum(1 for r in results if r['ground_truth'] == 0 and r['predicted'] == 0)
    fp = sum(1 for r in results if r['ground_truth'] == 0 and r['predicted'] == 1)
    fn = sum(1 for r in results if r['ground_truth'] == 1 and r['predicted'] == 0)
    
    print(f"\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.4f}")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    main()