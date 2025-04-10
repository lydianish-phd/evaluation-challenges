import os, argparse, json, yaml
import itertools
from comet import download_model, load_from_checkpoint
from utils import write_json

std_sentences = {
    "en": ["He's too funny!", "He's really funny!"],
    "fr": ["Il est trop drôle !", "Il est vraiment drôle !"]
}

ugc_sentences = {
    "en": ["his TOOOO funny!!", "he's rly FUNNYYYY!!!!!"],
    "fr": ["il est TROOOP drôle !!", "il et vraimnt DROOOOOLE !"]
}

sources = []
references = []
outputs = []
for src_lang, tgt_lang in itertools.permutations(["en", "fr"], 2):
    for src in ugc_sentences[src_lang]:
        for ref, output in itertools.permutations(std_sentences[tgt_lang] + ugc_sentences[tgt_lang], 2):
            sources.append(src)
            references.append(ref)
            outputs.append(output)

comet_model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_model_path)
xcomet_model_path = download_model("Unbabel/XCOMET-XL")
xcomet_model = load_from_checkpoint(xcomet_model_path)

data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, outputs, references)]

scores = {
    "comet": comet_model.predict(data, batch_size=32, gpus=1),
    "xcomet": xcomet_model.predict(data, batch_size=32, gpus=1)
}

output_dir = f"{os.environ['DATASETS']}/evaluation-challenges/experiment_049/analysis"
os.makedirs(output_dir, exist_ok=True)

write_json(f"{output_dir}/comet_scores.json", scores["comet"])



