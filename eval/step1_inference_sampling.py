import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from together import Together
import os

import json
import argparse
import random

import logging
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir',  type=str)
parser.add_argument('--n', default=32, type=int)
parser.add_argument('--sample_path', type=str, default=None)

args = parser.parse_args()
# Initialize an empty list to hold the dictionaries
data_list = []
    
CACHE_FILE = args.sample_path
    
def save_outputs(cached_outputs):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cached_outputs, f, indent=4)

def load_cached_outputs():
    """Load previously saved model outputs if they exist."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                cached_data = json.load(f)
                if isinstance(cached_data, list):
                    return cached_data  # Ensure it's a list
                else:
                    logging.warning("Cache file content is not a list. Resetting cache.")
                    return []
            except json.JSONDecodeError:
                logging.warning("Cache file is corrupted. Resetting cache.")
                return []
    return []  # Default to empty list

data_list = load_cached_outputs()

to_inference_codes = []
for i in range(len(data_list)):
    len_outputs = len(data_list[i]["model_outputs"])
    if len_outputs > args.n:
        assert len_outputs == len(data_list[i]["full_code"])
        indices = random.sample(range(len_outputs), args.n)
        data_list[i]["model_outputs"] = [data_list[i]["model_outputs"][index] for index in indices]
        data_list[i]["full_code"] = [data_list[i]["full_code"][index] for index in indices]
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

