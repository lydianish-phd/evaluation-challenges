import os, argparse, yaml
from openai import OpenAI
from prompt_templates import (
    GPT_MODEL_NAME,
    get_prompt
)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-file", type=str)
	parser.add_argument("-l", "--target-lang", type=str)
	parser.add_argument("-m", "--model-name", type=str, default=GPT_MODEL_NAME)
	args = parser.parse_args()

	client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

	with open(args.input_file, "r") as f:
        # ensure there are no trailing newlines which might affect the output
		sentences = [ line.strip() for line in f ]
	
	outputs = []
	for sentence in sentences:
		completion = client.chat.completions.create(
			model=GPT_MODEL_NAME,
			messages=get_prompt(sentence, args.target_lang, normalization=True, model_name=args.model_name),
			temperature=0, 
			top_p=1,
			max_tokens=512,
		)
		outputs.append(completion.choices[0].message.content.strip().replace("\n", " "))

	output_file = f"{args.input_file}.gpt"
	with open(output_file, "w") as f:
		for output in outputs:
			generated_text = output.outputs[0].text.strip().replace("\n", " ")
			f.write(f"{generated_text}\n")

	print(f" - Normalized sentences saved to {output_file}")
