import os, argparse, json, yaml
from sacrebleu.metrics import BLEU, CHRF
from comet import download_model, load_from_checkpoint
from prompt_templates import GUIDELINE_NAMES

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA = "google/gemma-2-9b-it"
NLLB = "facebook/nllb-200-3.3B"
CORPORA_CONFIG = os.path.join(os.environ["HOME"], "evaluation-challenges/src/llm/config/corpora.yaml")

def get_files(corpora, models, guidelines, output_dir, corpora_config=CORPORA_CONFIG):
    with open(corpora_config, "r") as f:
        config = yaml.safe_load(f)
    files = []
    for corpus in corpora:
        src_file = os.path.expandvars(config[corpus]["src_file_path"])
        ref_file = os.path.expandvars(config[corpus]["ref_file_path"])
        src_file_name = os.path.basename(src_file)
        sys_files = []
        
        for model in models:
            src_file_prefix = os.path.join(output_dir, model, corpus)
            if model == NLLB:
                sys_files.append(os.path.join(src_file_prefix, f"{src_file_name}.out"))
            else:
                for guideline in guidelines:
                    sys_files.append(os.path.join(src_file_prefix, f"{src_file_name}.{guideline}.out"))
        
        files.append((src_file, ref_file, sys_files))
    
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=str, nargs="+", default=["rocsmt", "footweets", "mmtc", "pfsmb"])
    parser.add_argument("--models", type=str, nargs="+", default=[LLAMA, GEMMA, NLLB])
    parser.add_argument("--guidelines", type=str, nargs="+", default=GUIDELINE_NAMES)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument("--overwrite", help="whether to overwrite existing output files", type=bool, default=False, action="store_true")
    args = parser.parse_args()

    print("Loading metric models...")
    bleu_model = BLEU()
    chrf_model = CHRF(word_order=2) # chrf++
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    xcomet_model_path = download_model("Unbabel/XCOMET-XL")
    xcomet_model = load_from_checkpoint(xcomet_model_path)
    
    files = get_files(args.corpora, args.models, args.guidelines, args.output_dir, args.corpora_config)
    
    for (src_file, ref_file, sys_files) in files:
        print(f"Evaluating outputs for {src_file} and {ref_file}...")
        
        print(" - Loading data...")
        with open (src_file) as f:
            src_data = [ line.strip() for line in f.readlines() ]

        with open (ref_file) as f:
            ref_data = [ line.strip() for line in f.readlines() ]

        print(" - Computing BLEU, ChrF++ and COMET scores...")
        for sys_file in sys_files:
            scores_file = f"{sys_file}.scores.json"
            errors_file = f"{sys_file}.errors.json"
            if not args.overwrite and os.path.exists(scores_file) and os.path.exists(errors_file):
                print(f" - Skipping {sys_file} as scores and errors already exist.")
                continue
            with open (sys_file) as f:
                sys_data = [ line.strip() for line in f.readlines() ]
            data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_data, sys_data, ref_data)]
    
            scores = {
                "bleu": bleu_model.corpus_score(sys_data, [ref_data]).score,
                "chrf2": chrf_model.corpus_score(sys_data, [ref_data]).score,
                "comet": comet_model.predict(data, batch_size=32, gpus=1)[1]
            }
            xcomet_output = xcomet_model.predict(data, batch_size=32, gpus=1)
            scores["xcomet"] = xcomet_output.system_score

            errors = []
            for score, span in zip(xcomet_output.scores, xcomet_output.metadata.error_spans):
                errors.append({
                    "score": score,
                    "span": span
                })
        
            with open(scores_file, 'w') as f:
                json.dump(scores, f)

            with open(errors_file, 'w') as f:
                json.dump(errors, f)

