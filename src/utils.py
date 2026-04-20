# Description: Contains main constants and utility functions.

import json, yaml, os
from pathlib import Path


GREEDY_CONFIG = (
    Path(__file__).resolve().parent / "config" / "greedy.yaml"
)

CORPORA_CONFIG = (
    Path(__file__).resolve().parent / "config" / "corpora.yaml"
)


ROCSMT = "rocsmt"
FOOTWEETS = "footweets"
MMTC = "mmtc"
PFSMB = "pfsmb"

CORPORA = [ROCSMT, FOOTWEETS, MMTC, PFSMB]
CORPUS_LABELS = {
    ROCSMT: "RoCS-MT",
    FOOTWEETS: "FooTweets",
    MMTC: "MMTC",
    PFSMB: "PFSMB",
}

TOWER = "Unbabel/TowerInstruct-7B-v0.2"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA = "google/gemma-2-9b-it"
NLLB = "facebook/nllb-200-3.3B"
GPT = "gpt-4o-mini"

MODEL_LABELS = {
    NLLB: "NLLB-3B",
    LLAMA: "Llama-3.1-8B",
    GEMMA: "Gemma-2-9B",
    TOWER: "Tower-7B-v0.2",
}

BLEU = "bleu"
CHRF = "chrf2"
COMET = "comet"
COMETKIWI = "cometkiwi"
XCOMET = "xcomet"

METRIC_LABELS = {
    BLEU: "BLEU",
    CHRF: "ChrF++",
    COMET: "COMET",
    COMETKIWI: "COMET-Kiwi",
    XCOMET: "xCOMET-XL",
}

def get_model_name(full_name):
    """
    Extracts the model name from the full model identifier.
    
    args:
        full_name: str, full model identifier
    returns:
        str, model name
    """
    return full_name.split("/")[-1]


def read_file(file):
    """
    args:
        file: str, path to file
    returns:
        list of str, lines of file (stripped)
    """
    with open(file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]
    
def read_json(file):
    with open(file, "r") as f:
        return json.load(f)

def write_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def read_yaml(file):
    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_config(config_path: str, data_dir: str | None = None):
    config = read_yaml(config_path)

    if data_dir is None:
        return config

    data_dir = Path(data_dir)

    resolved_config = {}
    for corpus, corpus_cfg in config.items():
        corpus_cfg = dict(corpus_cfg)

        for key in ["src_file_path", "ref_file_path"]:
            if key in corpus_cfg:
                path = Path(corpus_cfg[key])
                if not path.is_absolute():
                    path = data_dir / path
                corpus_cfg[key] = str(path)

        resolved_config[corpus] = corpus_cfg

    return resolved_config