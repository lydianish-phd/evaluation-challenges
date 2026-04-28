# Description: Contains main constants and utility functions.

import json, yaml, os, re
from pathlib import Path
from typing import Sequence
from .constants import MODEL_LABELS

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

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def write_lines(path: str, lines: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

def read_json(file):
    with open(file, "r") as f:
        return json.load(f)

def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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

        for key in ["src_file_path", "ref_file_path", "ugc_annotations_file_path", "norm_file_path"]:
            if key in corpus_cfg:
                path = Path(corpus_cfg[key])
                if not path.is_absolute():
                    path = data_dir / path
                corpus_cfg[key] = str(path)

        resolved_config[corpus] = corpus_cfg

    return resolved_config

def extract_guideline(file_name: str) -> str:
    match = re.search(r"\.(default|footweets|mmtc|pfsmb|rocsmt)\.out\.postproc$", file_name)
    if match:
        return match.group(1)
    return "baseline"

def sanitize_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model).replace(" ", "_").replace("/", "_")
