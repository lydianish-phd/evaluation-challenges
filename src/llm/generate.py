import os, argparse, yaml
from vllm import LLM, SamplingParams

LLAMA_DIR = os.path.join(os.environ["DSDIR"], "HuggingFace_Models/meta-llama/Meta-Llama-3.1-8B-Instruct")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=LLAMA_DIR)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--config-file", type=str, default="./config/llama.yaml")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    llm = LLM(model=args.model, )
    sampling_params = SamplingParams(
        n=config["n"],
        best_of=config["best_of"],
        temperature=config["temperature"], 
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        seed=config["seed"],
    )
    
    prompts = [
        "Translate to French:\n his TOOOO funny!!",
        "Translate to French. Give answer only.:\n his TOOOO funny!!",
        "Translate to standard French. Give answer only.:\n his TOOOO funny!!",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
