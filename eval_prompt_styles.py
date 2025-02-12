import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run eval.py with specified parameters.")
    parser.add_argument("--model_name", required=True, help="Name of the model")
    
    args = parser.parse_args()
    
    datasets = ["proofnet", "minif2f"]
    prompt_styles = ["think", "comments", "plain", "no_comments", "few_shot_no_comments"]
    
    for dataset in datasets:
        for prompt_style in prompt_styles:
            logging.info(f"Running eval.py with dataset {dataset}, prompt style {prompt_style}")
            command = (
                f"python eval/eval_local.py "
                f"--model_name {args.model_name} "
                f"--dataset {dataset} "
                f"--num_sampling 32 "
                f"--ngpu 2 "
                f"--prompt_style {prompt_style}"
            )
            os.system(command)

if __name__ == "__main__":
    main()

