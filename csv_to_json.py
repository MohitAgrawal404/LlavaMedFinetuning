"""
This is how you convert the csv binary classification to the vqa json
"""

import csv
import json
import uuid

def csv_to_json(csv_path, json_path):
    data = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Determine if Wilms tumor is present (1 or 0 as int)
            wilms = int(row['wilms_tumor'])
            # Compose GPT answer based on wilms_tumor value
            gpt_answer = ("Yes, this image shows a Wilms tumor."
                          if wilms == 1 else
                          "No, this is a ct scan of a different disease.")
            
            entry = {
                "id": str(uuid.uuid4()),
                "image": f"Wilms_Tumor/train/{row['filename']}",
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDoes this image show a Wilms tumor?"
                    },
                    {
                        "from": "gpt",
                        "value": gpt_answer
                    }
                ]
            }
            data.append(entry)

    # Write JSON output
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    csv_input_path = "./Wilms_Tumor/_classes_valid.csv"  # replace with your CSV filename
    json_output_path = "processed_valid_data.json"
    csv_to_json(csv_input_path, json_output_path)
    print(f"Converted {csv_input_path} to {json_output_path}")
