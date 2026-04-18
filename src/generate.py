import os, argparse, yaml
from vllm import LLM, SamplingParams
from .prompt_templates import (
    get_prompt,
    GUIDELINES
)
from .utils import read_config, read_file, GREEDY_CONFIG
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Path to the input file.")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("-m", "--model-dir", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("-c", "--config-file", type=str, default=GREEDY_CONFIG)
    parser.add_argument("-d", "--data-dir", type=str, help="Path to the data directory.")
    parser.add_argument("--source-lang", type=str, default="English")
    parser.add_argument("--target-lang", type=str, default="French")
    parser.add_argument("-g", "--guidelines", type=str, nargs="+", default=["default"])
    parser.add_argument("--overwrite", help="whether to overwrite existing output files", default=False, action="store_true")    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    model_name = os.path.basename(args.model_dir)

    config = read_config(args.config_file, args.data_dir)

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
        
    sentences = read_file(args.input_file)

    for guideline in args.guidelines:
        if guideline not in GUIDELINES:
            raise ValueError(f"Invalid guideline: {guideline}, expected one of {GUIDELINES.keys()}.")
        
        output_file = os.path.join(args.output_dir, f"{file_name}.{guideline}.out")
        
        if not args.overwrite and os.path.exists(output_file):
            print(f" - Skipping {output_file}")
            continue
        
        print(f" - Generating translations with the {guideline} guidelines...")

        prompts = [ get_prompt(sentence, args.source_lang, args.target_lang, model_name=model_name, guidelines=guideline) for sentence in sentences ]
        outputs = llm.generate(prompts, sampling_params)

        with open(output_file, "w") as f:
            for output in outputs:
                generated_text = output.outputs[0].text.strip().replace("\n", " ")
                f.write(f"{generated_text}\n")

        print(f" - Output translations saved to {output_file}")