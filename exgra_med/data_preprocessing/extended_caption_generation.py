import json
from datetime import datetime
from tqdm import tqdm
import os
import argparse
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def query_llm(
    system_prompt: str, user_prompt: str, model_name: str = "openai/gpt-4o-mini"
):
    """
    Query the LLM with the given system and user prompts.
    Args:
        system_prompt (str): The system prompt to set the context.
        user_prompt (str): The user prompt containing the instruction and answer.
        model_name (str): The LLM model to use.
    Returns:
        Optional[str]: The revised answer from the LLM, or None if parsing fails.
    """
    client = OpenAI(
        base_url=os.getenv("OPENROUTER_ENDPOINT"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    completion = client.chat.completions.create(
        extra_body={}, model=model_name, messages=messages
    )
    try:
        content = json.loads(completion.choices[0].message.content)
        revision = parse_llm_output(content)
        return revision
    except Exception as e:
        print("Error parsing LLM response:", e)
        return None


def load_system_prompt(fpath: str) -> str:
    with open(fpath, "r") as f:
        return f.read()


def detect_conversations_key(ele: Dict) -> Optional[str]:
    if "conversatons" in ele:
        return "conversatons"
    if "conversations" in ele:
        return "conversations"
    return None


def parse_llm_output(output: Any) -> Optional[str]:
    """
    Parse the LLM output to extract the revised answer.
    Args:
        output (Any): The output from the LLM, expected to be a dict or string.
    Returns:
        Optional[str]: The revised answer if found, else None.
    """
    # Expected: dict with 'revision', but sometimes string; try robust extraction.
    if isinstance(output, dict) and "revision" in output:
        return output["revision"]
    if isinstance(output, str):
        # try to find "revision": "<value>" or 'revision': '<value>'
        for marker in ['"revision":', "'revision':"]:
            if marker in output:
                try:
                    # crude extraction: take substring after marker and parse quotes
                    right = output.split(marker, 1)[1].strip()
                    # remove enclosing braces/trailing comma
                    if right.startswith(":"):
                        right = right[1:].strip()
                    # find the first quote
                    first_quote = right.find('"')
                    if first_quote == -1:
                        first_quote = right.find("'")
                    if first_quote != -1:
                        quote_char = right[first_quote]
                        # find matching end quote
                        rest = right[first_quote + 1 :]
                        end_idx = rest.find(quote_char)
                        if end_idx != -1:
                            return rest[:end_idx]
                except Exception:
                    continue
    return None


def run_extended_caption_generation(
    original_instruction_fpath: str,
    system_prompt: str,
    resume_from: str = None,
    model_name: str = "openai/gpt-4o-mini",
):
    """
    Run the extended caption generation process.
    Args:
        original_instruction_fpath (str): Path to the original instruction JSON file.
        system_prompt (str): The system prompt to use for the LLM.
        resume_from (str): Path to resume from an existing extended instruction file.
        model_name (str): The LLM model name to use.
    """
    llm_model_name = model_name.split("/")[-1]
    print(f"Using LLM model: {llm_model_name}")
    if resume_from and os.path.exists(resume_from):
        extended_instruction_fpath = resume_from
        ## Load existing extended instructions to continue
        print(f"Resuming from existing file: {extended_instruction_fpath}")
        extended_instructions = json.load(open(extended_instruction_fpath, "r"))
        extended_ids = [ele["id"] for ele in extended_instructions]
        ## Determine the ids needed to be processed
        original_instructions = json.load(open(original_instruction_fpath, "r"))
        instructions_to_process = [
            ele for ele in original_instructions if ele["id"] not in extended_ids
        ]
        print(
            f"Loaded {len(extended_ids)} existing extended instructions. There are {len(original_instructions)} original instructions."
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extended_instruction_fpath = original_instruction_fpath.replace(
            ".json", f"_extended_{llm_model_name}_{timestamp}.json"
        )
        instructions_to_process = json.load(open(original_instruction_fpath, "r"))
        print(
            f"Starting fresh. Will save extended instructions to: {extended_instruction_fpath}"
        )
        extended_instructions = []

    for ele in tqdm(instructions_to_process):
        conversation_key = detect_conversations_key(ele)
        if conversation_key:
            conversation = ele[conversation_key]
            new_con = []
            question = None
            for idx, role in enumerate(conversation):
                if idx % 2 == 0:
                    new_con.append(role)
                    question = role.get("value")
                else:
                    instruction = question
                    answer = role.get("value", "")
                    user_prompt = json.dumps(
                        {"instruction": instruction, "answer": answer}
                    )
                    while True:
                        revision = query_llm(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            model_name=model_name,
                        )
                        if revision is not None:
                            break
                        else:
                            print("Retrying LLM query for element id:", ele.get("id"))
                    role["value"] = revision
                    new_con.append(role)
            ele[conversation_key] = new_con
            ## Append to the extended instructions file to JSON and dump it
            extended_instructions.append(ele)
            with open(extended_instruction_fpath, "w") as f:
                json.dump(extended_instructions, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Extended Caption Generation using LLMs"
    )
    parser.add_argument(
        "--original_instruction_fpath",
        type=str,
        required=False,
        help="Path to the original instruction JSON file",
    )
    parser.add_argument(
        "--system_prompt_fpath",
        type=str,
        required=False,
        help="Path to the system prompt text file",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to resume from an existing extended instruction file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-4o-mini",
        help="LLM model name to use",
    )
    args = parser.parse_args()
    system_prompt = load_system_prompt(args.system_prompt_fpath)
    run_extended_caption_generation(
        original_instruction_fpath=args.original_instruction_fpath,
        system_prompt=system_prompt,
        resume_from=args.resume_from,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
