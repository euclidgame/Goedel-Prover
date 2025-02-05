import argparse
import os
import subprocess

def run_command(command):
    """Runs a shell command and ensures it executes successfully."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed: {command}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run model with dataset and sampling options.")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--dataset_name", type=str, default="minif2f", help="Name of the dataset.")
    parser.add_argument("--num_sampling", type=int, required=True, help="Number of sampling iterations.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output files.")
    parser.add_argument("--ngpu", type=int, default=2, help="Number of GPUs to use.")
    parser.add_argument("--ncpu", type=int, default=128, help="Number of CPUs to use.")
    parser.add_argument("--split", type=str, default="test", help="Split to use.")
    parser.add_argument("--field", type=str, default="complete", help="Field to use.")
    parser.add_argument("--subset", type=int, default=None, help="Subset to use.")
    parser.add_argument("--together", type=bool, default=False, help="Use together or not.")
    parser.add_argument("--sample_path", type=str, default=None, help="Path to the sample file.")
    parser.add_argument("--prompt_style", type=str, default=None, help="Prompt style to use.")
    args = parser.parse_args()

    dataset_path = f"datasets/{args.dataset_name}.jsonl"

    # Auto-generate output_dir if not provided
    if args.output_dir is None:
        if args.prompt_style == "plain":
            args.output_dir = f"results/pass_at_{args.num_sampling}/{args.dataset_name}/{args.model_name}/{args.prompt_style}"
        else:
            args.output_dir = f"results/pass_at_{args.num_sampling}/{args.dataset_name}/{args.model_name}"

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using output directory: {args.output_dir}")

    if args.sample_path is not None:
        inference_command = (
            f"python eval/step1_inference_sampling.py --output_dir {args.output_dir} --sample_path {args.sample_path} --n {args.num_sampling}"
        )
        run_command(inference_command)
    elif args.together:
        inference_command = (
            f"python eval/step1_inference_deepseek.py --input_path {dataset_path} "
            f"--model_path {args.model_name} --output_dir {args.output_dir} "
            f"--split {args.split} --n {args.num_sampling} --gpu {args.ngpu}"
        )
        if args.subset is not None:
            inference_command += f" --subset {args.subset}"
        run_command(inference_command)  
    else:
        inference_command = (
            f"python eval/step1_inference.py --input_path {dataset_path} "
            f"--model_path {args.model_name} --output_dir {args.output_dir} "
            f"--split {args.split} --n {args.num_sampling} --gpu {args.ngpu}"
        )
        if args.subset is not None:
            inference_command += f" --subset {args.subset}"
        if args.prompt_style is not None:
            inference_command += f" --prompt_style {args.prompt_style}"
        run_command(inference_command)

    compile_command = (
        f"python eval/step2_compile.py --input_path {args.output_dir}/to_inference_codes.json "
        f"--output_path {args.output_dir}/code_compilation.json --cpu {args.ncpu}"
    )
    run_command(compile_command)
    
    summarize_command = (
        f"python eval/step3_summarize_compile.py --input_path {args.output_dir}/code_compilation.json "
        f"--output_path {args.output_dir}/compilation_summarize.json --field {args.field}"
    )
    run_command(summarize_command)


if __name__ == "__main__":
    main()