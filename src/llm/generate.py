import os, argparse, yaml
from vllm import LLM, SamplingParams
from prompt_templates import (
    get_prompt,
    LLAMA_MODEL_NAME,
    GUIDELINES
)
import torch

LLAMA_DIR = os.path.join(os.environ["MODELS"], f"meta-llama/{LLAMA_MODEL_NAME}")
GREEDY_CONFIG = os.path.join(os.environ["HOME"], "evaluation-challenges/src/llm/config/greedy.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, default="")
    parser.add_argument("-o", "--output-dir", type=str, default="")
    parser.add_argument("-m", "--model-dir", type=str, default=LLAMA_DIR)
    parser.add_argument("-c", "--config-file", type=str, default=GREEDY_CONFIG)
    parser.add_argument("-l", "--target-lang", type=str, default="French")
    parser.add_argument("-g", "--guidelines", type=str, nargs="+", default=["default"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    model_name = os.path.basename(args.model_dir)

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    llm = LLM(
        model=args.model_dir, 
        max_model_len=config["max_model_len"],
        dtype=torch.bfloat16,
    )
    sampling_params = SamplingParams(
        n=config["n"],
        best_of=config["best_of"],
        temperature=config["temperature"], 
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
    )
        
    with open(args.input_file, "r") as f:
        # ensure there are no trailing newlines which might affect the output
        sentences = [ line.strip() for line in f ] 

    for guideline in args.guidelines:
        if guideline not in GUIDELINES:
            raise ValueError(f"Invalid guideline: {guideline}, expected one of {GUIDELINES.keys()}.")
        
        print(f" - Generating translations with the {guideline} guidelines...")

        prompts = [ get_prompt(sentence, args.target_lang, model_name=model_name, guidelines=guideline) for sentence in sentences ]
        outputs = llm.generate(prompts, sampling_params)

        output_file = os.path.join(args.output_dir, f"{file_name}.{guideline}.out")
        with open(output_file, "w") as f:
            for output in outputs:
                generated_text = output.outputs[0].text.strip().replace("\n", " ")
                f.write(f"{generated_text}\n")

        print(f" - Output translations saved to {output_file}")