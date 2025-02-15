import json
import argparse
from transformers import AutoTokenizer

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def truncate_text(text):
    return text.split("```", 1)[0]

# Calculate token usage
def calculate_average_token_usage(data, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_tokens = 0
    count = 0

    for entry in data:
        if "token_usage" in entry:
            total_tokens += entry["token_usage"]
        elif "model_outputs" in entry and isinstance(entry["model_outputs"], list):
            token_counts = [
                len(tokenizer(truncate_text(text), add_special_tokens=True)["input_ids"])
                for text in entry["model_outputs"]
            ]
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            total_tokens += avg_tokens

        count += 1

    return total_tokens / count if count > 0 else 0

# Main function
def main():
    parser = argparse.ArgumentParser(description="Calculate average token usage from JSON data.")
    parser.add_argument("file_path", type=str, help="Path to the JSON file.")
    parser.add_argument("model_name", type=str, help="Model name for tokenization.")
    args = parser.parse_args()

    data = load_json(args.file_path)
    avg_token_usage = calculate_average_token_usage(data, args.model_name)
    print(f"Average token usage: {avg_token_usage:.2f}")

if __name__ == "__main__":
    main()