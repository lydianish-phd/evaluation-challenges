# Description: Contains main constants and utility functions.

import json, yaml, os
from pathlib import Path

TOWER = "Unbabel/TowerInstruct-7B-v0.2"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA = "google/gemma-2-9b-it"
NLLB = "facebook/nllb-200-3.3B"
GPT = "gpt-4o-mini"

GREEDY_CONFIG = (
    Path(__file__).resolve().parent / "config" / "greedy.yaml"
)

CORPORA_CONFIG = (
    Path(__file__).resolve().parent / "config" / "corpora.yaml"
)

CORPORA = ["rocsmt", "footweets", "mmtc", "pfsmb"]

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
    with open(file, "r") as f:
        return [line.strip() for line in f]
    
def read_json(file):
    with open(file, "r") as f:
        return json.load(f)

def write_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

def read_yaml(file):
    with open(file, "r") as f:
        return yaml.safe_load(f)