#!/bin/bash
set -e

commands=(
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -o results/minif2f/DeepSeek-R1-Distill-Qwen-32B-Thinking -n 32 -g 4 -c 128"
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -o results/minif2f/pass_8/DeepSeek-R1-Distill-Qwen-32B-Thinking -n 8 -g 4 -c 128"
    "git checkout main"
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m Qwen/Qwen2.5-32B -o results/minif2f/pass_8/Qwen-32B -n 8 -g 4 -c 128"
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m Qwen/QwQ-32B-Preview -o results/minif2f/pass_8/QwQ-32B -n 8 -g 4 -c 128"
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m Goedel-LM/Goedel-Prover-SFT -o results/minif2f/pass_8/Godel-Prover-SFT -n 8 -g 4 -c 128"
    "sh eval/eval.sh -i datasets/minif2f.jsonl -s test -m deepseek-ai/DeepSeek-Prover-V1.5-RL -o results/minif2f/pass_8/DSProver -n 8 -g 4 -c 128"
)

for command in "${commands[@]}"; do
    echo "Running: $command"
    bash -c "$command" | tee -a script_output.log
done

echo "All tasks completed!"


