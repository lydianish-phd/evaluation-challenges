import os, argparse, yaml
from vllm import LLM, SamplingParams
from prompt_templates import (
    get_prompt,
    LLAMA_MODEL_NAME,
)

LLAMA_DIR = os.path.join(os.environ["DSDIR"], "HuggingFace_Models/meta-llama")
CONFIG_DIR = os.path.join(os.environ["HOME"], "evaluation-challenges/src/llm/config")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=LLAMA_DIR)
    parser.add_argument("--model-name", type=str, choices=[LLAMA_MODEL_NAME], default=LLAMA_MODEL_NAME)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--config-file", type=str, default=f"{CONFIG_DIR}/llama.yaml")
    parser.add_argument("--target-lang", type=str, default="French")
    parser.add_argument("--guidelines", type=str, choices=["none", "standard", "general"], default="none")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = os.path.basename(args.input_file)
    output_file = os.path.join(args.output_dir, f"{file_name}.out")
    
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_path = os.path.join(args.model_dir, args.model_name)
    llm = LLM(model=model_path, max_num_batched_tokens=config["max_tokens"])
    sampling_params = SamplingParams(
        n=config["n"],
        best_of=config["best_of"],
        temperature=config["temperature"], 
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
    )
    
    if args.seed:
        sampling_params.seed = args.seed

    with open(args.input_file, "r") as f:
        sentences = f.readlines()

    prompts = [ get_prompt(sentence, args.target_lang, args.model_type, args.guidelines) for sentence in sentences ]
    outputs = llm.generate(prompts, sampling_params)
    
    with open(output_file, "w") as f:
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            f.write(f"{generated_text}\n")

    print(f"Generated translations saved to {output_file}")