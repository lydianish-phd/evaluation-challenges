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

DEFAULT = "default"
STANDARD = "standard"
GENERAL = "general"

GUIDELINE_LABELS = {
    DEFAULT: "None",
    STANDARD: "Standard",
    GENERAL: "+General",
    ROCSMT: "+RoCS-MT",
    FOOTWEETS: "+FooTweets",
    MMTC: "+MMTC",
    PFSMB: "+PFSMB",
}

MINOR = "minor"
MAJOR = "major"
CRITICAL = "critical"