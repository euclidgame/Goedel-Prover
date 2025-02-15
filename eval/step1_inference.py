import os
import re
import json
import argparse
import logging
import concurrent.futures

from openai import OpenAI

from openai.types.completion_usage import CompletionUsage

logging.basicConfig(level=logging.INFO)

LEAN4_DEFAULT_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""

def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Lean 4 solutions via OpenAI API calls.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON lines file.")
    parser.add_argument("--model_path", type=str, required=True, help="The OpenAI model name to use (e.g., gpt-4).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store output JSON files.")
    parser.add_argument("--split", default="none", type=str, help="Split value used for filtering data.")
    parser.add_argument("--n", default=2, type=int, help="Number of completions per prompt.")
    parser.add_argument("--subset", type=int, default=None, help="Process only the first N records.")
    parser.add_argument("--prompt_style", default="comments", type=str, help="Style of prompt to generate.")
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="Base URL for custom OpenAI endpoints (or 'https://api.openai.com/v1' if standard).",
    )
    return parser.parse_args()


def load_data(data_path: str, split: str = "none", subset: int = None) -> list:
    """
    Load data from a JSON lines file and optionally filter by 'split' or truncate by 'subset'.
    Returns a list of data items (dicts).
    """
    data_list = []
    with open(data_path, "r") as file:
        for line in file:
            item = json.loads(line)
            if split == "none":
                data_list.append(item)
            else:
                # Attempt to interpret the split argument as an integer or string
                try:
                    int_split = int(split)
                    # Filter only if item["split"] matches the integer
                    if int(item.get("split", -1)) == int_split:
                        data_list.append(item)
                except ValueError:
                    # If not int, treat as string comparison
                    if item.get("split") == split:
                        data_list.append(item)

    if subset is not None:
        data_list = data_list[:subset]

    return data_list


def make_conversations(item: dict, prompt_style: str) -> list[dict]:
    """
    Build a list of messages (conversation) for the OpenAI Chat API.
    
    1. The user message instructs the LLM to complete the Lean 4 code.
    2. The assistant message either:
       - Starts with <think> (if prompt_style == "think"), or
       - Starts with a code block containing partial Lean code (otherwise).
    """
    header = item.get("header", LEAN4_DEFAULT_HEADER)
    informal_prefix = item.get("informal_prefix", "")
    formal_statement = item["formal_statement"]

    # This is the user prompt. It's the same across all prompt styles except for the text instructions,
    # which you can customize as needed.
    prompt_templates = {
        "think": (
            "Prove the following theorem in Lean 4 by completing the following Lean 4 code which contains a header, "
            "informal prefix, and formal statement."
            "Please reason step by step first and wrap your thinking process within <think> and </think> tags. "
            "Then, enclose your final code within a Lean 4 code block that starts with:\n```lean4\n{header}{informal_prefix}{formal_statement}\n```\n"
        ),
        "comments": (
            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n"
            "```lean4\n{header}{informal_prefix}{formal_statement}\n```\n"
            "Make sure your code successfully proves the formal statement."
        ),
        "plain": (
            "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}\n```"
        ),
        "no_comments": (
            "Complete the following Lean 4 code WITHOUT ANY comments or explanations in the code:\n\n"
            "```lean4\n{header}{informal_prefix}{formal_statement}\n```"
        ),
        "few_shot_no_comments": (
            "Complete the following Lean 4 code WITHOUT ANY comments or explanations in the code. "
            "That means your code should have a similar style as the following examples:\n\n"
            "Example 1:\n```lean4\ntheorem mathd_algebra_182 (y : ℤ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n"
            "  simp [mul_add, mul_comm, mul_left_comm]\n  ring_nf\n```\n\n"
            "Example 2:\n```lean4\ntheorem mathd_algebra_182 (y : ℤ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n"
            "  norm_num\n  ring\n  <;> linarith\n```\n\n"
            "Now, complete the following Lean 4 code WITHOUT ANY comments or explanations in the code:\n\n"
            "```lean4\n{header}{informal_prefix}{formal_statement}"
        ),
    }

    if prompt_style not in prompt_templates:
        raise ValueError(f"Invalid prompt style: {prompt_style}")

    user_content = prompt_templates[prompt_style].format(
        header=header,
        informal_prefix=informal_prefix,
        formal_statement=formal_statement,
    )

    # The assistant's initial message differs based on whether we're "thinking" or not.
    if prompt_style == "think":
        # Start with <think> only
        assistant_content = "<think>"
    else:
        # Start with a code block containing the partial code
        assistant_content = f"<think>\nOkay, I have finished thinking.\n</think>\nHere is my final code:\n```lean4\n{header}{informal_prefix}{formal_statement}"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def extract_code(llm_output: str, item: dict, prompt_style: str) -> str:
    """
    Attempt to parse out Lean code from a triple-backtick block in the LLM output.
    If not found, fall back to a default that includes the original statement.
    In 'few_shot_no_comments' style, we handle a special case to parse the 3rd code block.
    """
    # TODO (wenjie): check if the code contains necessary statements like the theorem name.
    header = item.get("header", LEAN4_DEFAULT_HEADER)
    formal_statement = item.get("formal_statement", "-- ERROR: No formal statement found.")

    try:
        if prompt_style == "few_shot_no_comments":
            # In some multi-block responses, the relevant code might be in the 3rd block
            matches = list(re.finditer(r"```(lean4|lean)\n(.*?)\n```", llm_output, re.DOTALL))
            if len(matches) >= 3:
                return matches[2].group(2).strip()

        match = re.search(r"```(lean4|lean)\n(.*?)\n```", llm_output, re.DOTALL)
        if match:
            code = match.group(2).strip()
            if code.startswith(header):
                return code
            else:
                return f"{header}\n\n{code}"

        # Fallback if we find nothing
        return f"{header}\n\n{formal_statement}\n"

    except Exception as e:
        logging.warning(f"Error in extract_code: {e}")
        return f"{header}\n\n{formal_statement}\n"


def load_cached_outputs(cache_file: str) -> list:
    """Load previously saved model outputs if they exist, else return an empty list."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                if isinstance(cached_data, list):
                    return cached_data
                else:
                    logging.warning("Cache file content is not a list. Resetting cache.")
                    return []
        except json.JSONDecodeError:
            logging.warning("Cache file is corrupted. Resetting cache.")
            return []
    return []


def save_outputs(cached_outputs: list, cache_file: str) -> None:
    """Save model outputs to a JSON file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cached_outputs, f, indent=4)


def call_model_openai(
    data_list: list,
    conversations: list[list[dict]],
    args: argparse.Namespace,
    cache_file: str
) -> tuple[list, list]:
    """
    Call the OpenAI chat model in parallel using the conversation messages 
    from make_conversations().
    
    Returns:
        model_outputs (list[list[str]]): 
            A list (same length as data_list) of lists of LLM output strings.
        usage_list (list[dict]): 
            A list of usage objects (one per item) or None if usage not available.
    """
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )

    cached_outputs = load_cached_outputs(cache_file)
    model_outputs = []
    usage_list = []

    def worker(i: int):
        """Single worker function for concurrency and caching."""
        item_name = data_list[i].get("name", f"item_{i}")
        # Check cache first
        for c in cached_outputs:
            # If the item name matches and we have enough responses, skip re-calling
            if c["name"] == item_name and len(c["model_outputs"]) == args.n:
                logging.info(f"Cache hit for {item_name}")
                return c["model_outputs"], c["token_usage"]

        # Not in cache or incomplete => call the model
        logging.info(f"Calling model for index {i}: {item_name}")
        messages = conversations[i]

        try:
            response = client.chat.completions.create(
                model=args.model_path,
                messages=messages,
                max_tokens=2048,
                n=args.n,
                temperature=0.6,
                extra_body={"continue_final_message": True, "add_generation_prompt": False},
            )
            outputs = [choice.message.content for choice in response.choices]
            usage_info = response.usage  # e.g., usage_info["completion_tokens"]
        except Exception as e:
            logging.error(f"Error in model call for index {i}: {e}")
            outputs = [None] * args.n
            usage_info = None

        # Update the cache
        found_in_cache = False
        for c in cached_outputs:
            if c["name"] == item_name:
                c["model_outputs"] = outputs
                c["token_usage"] = usage_info.completion_tokens
                found_in_cache = True
                break
        if not found_in_cache:
            cached_outputs.append({
                "name": item_name,
                "model_outputs": outputs,
                "token_usage": usage_info.completion_tokens,
            })

        save_outputs(cached_outputs, cache_file)
        return outputs, usage_info

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, range(len(data_list))))

    for (outputs, usage_info) in results:
        model_outputs.append(outputs)
        usage_list.append(usage_info)

    return model_outputs, usage_list


def main():
    args = get_args()

    # 1. Load data
    data_list = load_data(args.input_path, args.split, args.subset)

    # 2. Build conversation messages for each data item
    conversations = [make_conversations(item, args.prompt_style) for item in data_list]

    # 3. Call the OpenAI model (always using OpenAI, ignoring any litellm/vllm logic)
    cache_file = os.path.join(args.output_dir, "model_outputs.json")
    model_outputs, usage_list = call_model_openai(data_list, conversations, args, cache_file)

    # 4. Extract code and organize final results
    for i in range(len(data_list)):
        data_list[i]["conversation"] = conversations[i]  # so we know what we asked
        data_list[i]["model_outputs"] = model_outputs[i]

        full_code_list = []
        # TODO (wenjie): Consider the case that the the generation doesn't continue the final message.
        assistant_messages = ""
        for message in conversations[i]:
            if message["role"] == "assistant":
                assistant_messages += message["content"]
        for output in model_outputs[i]:
            code = extract_code(assistant_messages + output, data_list[i], args.prompt_style)
            full_code_list.append(code)

        data_list[i]["full_code"] = full_code_list

        usage_info = usage_list[i]
        if usage_info and isinstance(usage_info, CompletionUsage) and usage_info.completion_tokens:
            data_list[i]["token_usage"] = usage_info.completion_tokens / float(args.n)
        else:
            data_list[i]["token_usage"] = None

    # Construct a separate list for inference
    to_inference_codes = []
    for i, item in enumerate(data_list):
        name = item.get("problem_id") or item.get("name", f"item_{i}")
        for code in item["full_code"]:
            to_inference_codes.append({"name": name, "code": code})

    # 5. Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Full records
    full_records_path = os.path.join(args.output_dir, "full_records.json")
    with open(full_records_path, "w") as f:
        json.dump(data_list, f, indent=4)
    logging.info(f"Saved full records to {full_records_path}")

    # To-inference codes
    to_inference_path = os.path.join(args.output_dir, "to_inference_codes.json")
    with open(to_inference_path, "w") as f:
        json.dump(to_inference_codes, f, indent=4)
    logging.info(f"Saved to-inference codes to {to_inference_path}")

    # Show average token usage if available
    valid_usages = [d.get("token_usage") for d in data_list if d.get("token_usage") is not None]
    if valid_usages:
        average_usage = sum(valid_usages) / len(valid_usages)
        logging.info(f"Average token usage across items: {average_usage:.2f}")
        with open(os.path.join(args.output_dir, "token_usage_summary.json"), "w") as f:
            json.dump({"average_usage": average_usage}, f, indent=4)
    else:
        logging.info("No token usage data to summarize.")


if __name__ == "__main__":
    main()