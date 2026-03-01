import os, argparse, json, yaml
from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint
from .prompt_templates import REFUSAL_TO_TRANSLATE
from .utils import (
    read_file, 
    read_yaml, 
    write_json,
    TOWER,
    LLAMA,
    GEMMA,
    NLLB,
    CORPORA_CONFIG,
    CORPORA
)
import numpy as np


def get_files(corpora, models, guidelines, output_dir, corpora_config=CORPORA_CONFIG):
    config = read_yaml(corpora_config)
    files = []
    for corpus in corpora:
        src_file = os.path.expandvars(config[corpus]["src_file_path"])
        ref_file = os.path.expandvars(config[corpus]["ref_file_path"])
        src_file_name = os.path.basename(src_file)
        sys_files = []
        
        for model in models:
            src_file_prefix = os.path.join(output_dir, model, corpus)
            if model == NLLB:
                sys_files.append(os.path.join(src_file_prefix, f"{src_file_name}.out.postproc"))
            else:
                for guideline in guidelines:
                    sys_files.append(os.path.join(src_file_prefix, f"{src_file_name}.{guideline}.out.postproc"))
        
        files.append((src_file, ref_file, sys_files))
    
    return files

def set_comet_scores_to_zero_for_empty(sys_data, comet_scores):
    for i, sys in enumerate(sys_data):
        if not sys or sys == REFUSAL_TO_TRANSLATE:
            comet_scores[i] = 0
    return comet_scores

def compute_comet_scores(sys_data, comet_model):
    comet_output = comet_model.predict(data, batch_size=32, gpus=1)
    comet_scores = comet_output.scores
    comet_scores = set_comet_scores_to_zero_for_empty(sys_data, comet_scores)         
    return comet_scores

def load_comet_model(model_name="Unbabel/wmt22-comet-da"):
    comet_model_path = download_model(model_name)
    comet_model = load_from_checkpoint(comet_model_path)
    return comet_model

MINOR = "minor"
MAJOR = "major"
CRITICAL = "critical"

def count_error_types(errors):
    counts = {
        MINOR: 0,
        MAJOR: 0,
        CRITICAL: 0
    }
    for error in errors:
        for span in error["spans"]:
            counts[span["severity"]] += 1
    counts["total"] = sum(counts.values())
    return counts

def get_sentences_with_errors(errors, severity):
    sentence_ids = []
    for i, sentence in enumerate(errors):
        for span in sentence["spans"]:
            if span["severity"] == severity:
                sentence_ids.append(i)
                break
    return sentence_ids

def get_correct_sentences(errors):
    sentence_ids = []
    for i, sentence in enumerate(errors):
        if len(sentence["spans"]) == 0:
            sentence_ids.append(i)
    return sentence_ids

def get_counts(errors):
    counts = count_error_types(errors)
    for severity in [MINOR, MAJOR, CRITICAL]:
        counts[f"{severity}_sents_ids"] = get_sentences_with_errors(errors, severity)
        counts[f"{severity}_sents"] = len(counts[f"{severity}_sents_ids"])
    counts["correct_sents_ids"] = get_correct_sentences(errors)
    counts["correct_sents"] = len(counts["correct_sents_ids"])
    counts["total_sents"] = len(errors)
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=str, nargs="+", default=CORPORA)
    parser.add_argument("--models", type=str, nargs="+", default=[TOWER, LLAMA, GEMMA, NLLB])
    parser.add_argument("--guidelines", type=str, nargs="+", default=["default"])
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument("--overwrite", help="whether to overwrite existing score files", default=False, action="store_true")
    parser.add_argument("--xcomet", help="whether to use xCOMET-XL", default=False, action="store_true")
    args = parser.parse_args()

    print("Loading metric models: BLEU, ChrF++, COMET")
    bleu_model = BLEU()
    chrf_model = CHRF(word_order=2) # chrf++
    comet_model = load_comet_model("Unbabel/wmt22-comet-da")
    cometkiwi_model = load_comet_model("Unbabel/wmt22-cometkiwi-da")
    if args.xcomet:
        print("Loading xCOMET-XL model")
        xcomet_model = load_comet_model("Unbabel/XCOMET-XL")
    
    files = get_files(args.corpora, args.models, args.guidelines, args.output_dir, args.corpora_config)
    
    for (src_file, ref_file, sys_files) in files:
        print(f"Evaluating outputs for {src_file} and {ref_file}...")
        
        src_data = read_file(src_file)
        ref_data = read_file(ref_file)
        
        for sys_file in sys_files:
            scores_file = f"{sys_file}.scores.json"
            comet_file = f"{sys_file}.comet.json"

            if (not args.overwrite and 
                os.path.exists(scores_file) and 
                os.path.exists(comet_file)
            ):
                print(f" - Skipping {sys_file}")
                continue

            print(f" - Computing scores for {sys_file}")
            
            sys_data = read_file(sys_file)
            scores = {
                "bleu": bleu_model.corpus_score(sys_data, [ref_data]).score,
                "chrf2": chrf_model.corpus_score(sys_data, [ref_data]).score,
            }
            
            data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_data, sys_data, ref_data)]
            
            comet_scores = {}
            comet_scores["comet"] = compute_comet_scores(sys_data, comet_model)
            comet_scores["cometkiwi"] = compute_comet_scores(sys_data, cometkiwi_model)
            scores["comet"] = np.mean(comet_scores["comet"])
            scores["cometkiwi"] = np.mean(comet_scores["cometkiwi"])
            write_json(comet_file, comet_scores)

            if args.xcomet:
                print(f" - Computing xCOMET scores for {sys_file}")
                errors_file = f"{sys_file}.errors.json"
                counts_file = f"{sys_file}.counts.json"
                
                if (not args.overwrite and 
                    os.path.exists(errors_file) and 
                    os.path.exists(counts_file)
                ):
                    print(f" - Skipping {sys_file}")
                    continue
                

                xcomet_output = xcomet_model.predict(data, batch_size=32, gpus=1)
                scores["xcomet"] = xcomet_output.system_score
                
                errors = []
                for score, spans in zip(xcomet_output.scores, xcomet_output.metadata.error_spans):
                    errors.append({
                        "score": score,
                        "spans": spans
                    })
                counts = get_counts(errors)
                
                write_json(errors_file, errors)
                write_json(counts_file, counts)
            
            write_json(scores_file, scores)



