import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from litellm import completion
import os

import json
import argparse
import concurrent.futures

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

api_model_path = ['openai/gpt-4o', 'openai/o1', 'openai/o1-mini']


# parser.add_argument('--output_path', default="example_data/o1_sorried_output.json", type=str)
args = parser.parse_args()


# data_path = "/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/datasets/minif2f_sanity10.jsonl"
data_path = args.input_path
# Initialize an empty list to hold the dictionaries
data_list = []

# Open the file and read each line
with open(data_path, 'r') as file:
    for line in file:
        # Parse the JSON object and append it to the list
        # if data_split is not None and prob['split'] not in data_split:
        #     continue
        data = json.loads(line)
        # if (data["split"] == args.split) or (args.split == "none"):
        #     data_list.append(data)
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
for data in data_list:
        # model_inputs.append("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}\n/-{informal_prefix}-/ \n{formal_statement}".format(
        model_inputs.append("Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )

model_name = args.model_path
# model_name = "/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/models/Translator_Qwen2.5-Coder-32B_numina_sonnet_130K_translator_Epoch2_LR1e-4"

def extract_code(inputs):
    try:
        return re.search(r'```(lean4|lean)\n(.*?)\n```', inputs, re.DOTALL).group(2)
    except AttributeError:
        return inputs

if model_name in api_model_path:
    model_outputs = []
    def call_model(i):
        messages = [{"role": "user", "content": model_inputs[i]}]
        outputs = []
        for _ in range(len(data_list)):
            try:
                response = completion(
                    model=model_name,
                    messages=messages,
                    n=args.n,
                    temperature=1.0,
                )
                outputs.extend([choice.message.content for choice in response.choices])
            except Exception as e:
                print(f"Error in model call for index {i}: {e}")
                outputs.append(None)  # Placeholder for failed response

        return outputs
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(call_model, range(len(data_list))))

    model_outputs.extend(results)
    to_inference_codes = []
    for i in range(len(model_outputs)):
        data_list[i]["model_input"] = model_inputs[i]
        data_list[i]["model_outputs"] = model_outputs[i]
        data_list[i]["full_code"] = [extract_code(output) for output in model_outputs[i]]
        if "problem_id" in data_list[i]:
            to_inference_codes += [{"name": data_list[i]["problem_id"], "code": code} for code in data_list[i]["full_code"]]
        else:
            to_inference_codes += [{"name": data_list[i]["name"], "code": code} for code in data_list[i]["full_code"]]
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = LLM(model=model_name, max_num_batched_tokens=8192, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu)
    model = LLM(model=model_name, seed=1, trust_remote_code=True, swap_space=8, tensor_parallel_size=args.gpu, max_model_len=4096)


    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=2048,
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
        data_list[i]["full_code"] = [extract_code(model_inputs[i] + output.text) for output in model_outputs[i].outputs]
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
# for data
#     model_outputs[0].outputs[0].text
