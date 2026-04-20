import os, argparse, yaml, time
from openai import OpenAI
from .constants import GPT, DEFAULT
from .prompt_templates import get_prompt

from .utils import read_file

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-file", type=str)
	parser.add_argument("-l", "--target-lang", type=str)
	parser.add_argument("-m", "--model-name", type=str, default=GPT)
	parser.add_argument("-g", "--guidelines", type=str, default=DEFAULT)
	args = parser.parse_args()

	client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

	sentences = read_file(args.input_file)
	
	file_name = os.path.basename(args.input_file)
	output_file = os.path.join(os.path.dirname(args.input_file), f"gpt.{file_name}")
	
	print(f" - Loaded {len(sentences)} sentences...")
	start_time = time.time()
	n = 0
	with open(output_file, "w") as f:
		for sentence in sentences:
			completion = client.chat.completions.create(
				model=GPT,
				messages=get_prompt(sentence, args.target_lang, args.target_lang, normalization=True, model_name=args.model_name, guidelines=args.guidelines),
				temperature=0, 
				top_p=1,
				max_tokens=512,
			)
			output = completion.choices[0].message.content.strip().replace("\n", " ")
			f.write(f"{output}\n")
			n += 1
			if n % 10 == 0:
				print(f" - {n} done...")

	print(f" - Normalized sentences saved to {output_file}")
	print(f" - Normalization took {time.time() - start_time:.2f} seconds")
