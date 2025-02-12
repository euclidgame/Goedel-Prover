import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from litellm import completion
import os

import json
import argparse
import concurrent.futures
from openai import OpenAI
import logging
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
# /scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/datasets/minif2f.jsonl
parser.add_argument('--input_path',  type=str)
# /scratch/gpfs/yl7690/models/DeepSeek-Prover-V1.5-RL
parser.add_argument('--model_path', type=str)
# results/test
parser.add_argument('--output_dir',  type=str)
parser.add_argument('--split', default="none", type=str)
parser.add_argument('--n', default=32, type=int)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--subset', type=int, default=None)
parser.add_argument('--prompt_style', default="comments", type=str)

api_model_path = ['openai/gpt-4o', 'openai/o1', 'openai/o1-mini', 'openai/o3-mini-2025-01-31']

args = parser.parse_args()

data_path = args.input_path
# Initialize an empty list to hold the dictionaries
data_list = []

# Open the file and read each line
with open(data_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if args.split == "none":
            data_list.append(data)
        else:
            try:
                int_split = int(args.split)
            except:
                int_split = None
                pass
            if isinstance(int_split, int):
                if (int(data["split"]) == int(args.split)):
                    data_list.append(data)
            else:
                if ((data["split"]) == (args.split)):
                    data_list.append(data)

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

model_inputs = []
if args.subset is not None:
    data_list = data_list[:args.subset]
for data in data_list:
    if args.prompt_style == "think":
        model_inputs.append("Question:\nComplete the following Lean 4 code which contains header, informal prefix and formal statement. You will need to provide the proof for the formal statement. Please reason step by step first and wrap your thinking process within \"<think>\\n\\n</think>\" and enclose your final code within a Lean 4 code block which starts with: \n```lean4\n{header}{informal_prefix}{formal_statement}\n```\n\nAnswer:\n<think>\n".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )
    elif args.prompt_style == "comments":
        model_inputs.append("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )
    elif args.prompt_style == "plain":
        model_inputs.append("Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )
    elif args.prompt_style == "no_comments":
        model_inputs.append("Complete the following Lean 4 code WITHOUT ANY comments or explanations in the code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )
    else:
        raise ValueError(f"Invalid prompt style: {args.prompt_style}")

model_name = args.model_path

def extract_code(inputs, data):
    if not isinstance(inputs, str):
        logging.info(f"Inputs is not a string: {inputs}")
        return "{header}\n\n{formal_statement}\n".format(
            header=data.get('header', LEAN4_DEFAULT_HEADER), 
            formal_statement=data.get('formal_statement', "-- ERROR: No formal statement found.")
        )

    try:
        match = re.search(r'```(lean4|lean)\n(.*?)\n```', inputs, re.DOTALL)
        proof = data.get('formal_statement', "-- ERROR: No formal statement found.")
        if match:
            proof = match.group(2).strip()  # Extract and strip whitespace
            if proof.startswith("import"):
                return proof
        return "{header}\n\n{formal_statement}\n".format(
            header=data.get('header', LEAN4_DEFAULT_HEADER), 
            formal_statement=proof
        )

    except Exception as e:
        logging.info(f"Error in extract_code: {e}")
        # Catch unexpected errors and return an invalid Lean statement
        return "{header}\n\n{formal_statement}\n".format(
            header=data.get('header', LEAN4_DEFAULT_HEADER), 
            formal_statement=data.get('formal_statement', "-- ERROR: No formal statement found.")
        )
    
CACHE_FILE = f"{args.output_dir}/full_records.json"
    
def save_outputs(cached_outputs):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cached_outputs, f, indent=4)

model_outputs = []
model_name = args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LLM(model=model_name, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=16382)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95,
    n=args.n,
)

model_outputs = model.generate(
    model_inputs,
    sampling_params,
    use_tqdm=True,
)
assert len(model_outputs) == len(model_inputs)
to_inference_codes = []
for i in range(len(data_list)):
    data_list[i]["model_input"] = model_inputs[i]
    data_list[i]["model_outputs"] = [output.text for output in model_outputs[i].outputs]
    data_list[i]["full_code"] = [extract_code(output, data_list[i]) for output in data_list[i]["model_outputs"]]
    if "problem_id" in data_list[i]:
        to_inference_codes += [{"name": data_list[i]["problem_id"], "code": code} for code in data_list[i]["full_code"]]
    else:
        to_inference_codes += [{"name": data_list[i]["name"], "code": code} for code in data_list[i]["full_code"]]

os.makedirs(args.output_dir, exist_ok=True)

output_file_path = F'{args.output_dir}/full_records.json'
print(F"Outputing to {output_file_path}")
# Dump the list to a JSON file with indents
with open(output_file_path, 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

toinfer_file_path = F'{args.output_dir}/to_inference_codes.json'
print(F"Outputing to {toinfer_file_path}")
# Dump the list to a JSON file with indents
with open(toinfer_file_path, 'w') as json_file:
    json.dump(to_inference_codes, json_file, indent=4)

