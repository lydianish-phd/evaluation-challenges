import os, argparse
from sacrebleu.metrics import BLEU as bleu, CHRF as chrf
from .utils import read_file, write_json, read_config
from .constants import (
    TOWER,
    LLAMA,
    GEMMA,
    CORPORA,
    CORPORA_CONFIG,
)

OUTPUT_SUFFIX = "out.postproc"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-d", "--data-dir", help="parent directory containing all corpora files referenced in corpora.yaml", type=str)
    parser.add_argument("--corpora", type=str, nargs="+", default=CORPORA)
    parser.add_argument("--models", type=str, nargs="+", default=[TOWER, LLAMA, GEMMA])
    parser.add_argument("--guidelines", type=str, nargs="+", default=["default"] + CORPORA)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    args = parser.parse_args()

    print("Loading metric models: BLEU, ChrF++")
    bleu_model = bleu()
    chrf_model = chrf(word_order=2) # chrf++

    config = read_config(args.corpora_config, args.data_dir)

    for corpus in args.corpora:
        src_file = config[corpus]["src_file_path"]
        src_file_name = os.path.basename(src_file)

        for model in args.models:
            bleu_scores = {}
            chrf_scores = {}
            src_file_prefix = os.path.join(f"{args.input_dir}/outputs", model, corpus, src_file_name)
            for guideline1 in args.guidelines:
                bleu_scores[guideline1] = {}
                chrf_scores[guideline1] = {}
                for guideline2 in args.guidelines:
                    # if guideline1 == guideline2:
                    #     continue
                    print(f"Comparing {guideline1} and {guideline2} for {model} on {corpus}...")
                    sys_file1 = f"{src_file_prefix}.{guideline1}.{OUTPUT_SUFFIX}"
                    sys_file2 = f"{src_file_prefix}.{guideline2}.{OUTPUT_SUFFIX}"
                    if not os.path.exists(sys_file1) or not os.path.exists(sys_file2):
                        print(f" - Skipping {sys_file1} and {sys_file2} as one of them does not exist")
                        continue
                    sys_data1 = read_file(sys_file1)
                    sys_data2 = read_file(sys_file2)
                    bleu_scores[guideline1][guideline2] = bleu_model.corpus_score(sys_data1, [sys_data2]).score
                    chrf_scores[guideline1][guideline2] = chrf_model.corpus_score(sys_data1, [sys_data2]).score
    

            # Write scores to JSON files
            output_dir = f"{args.input_dir}/output_comparisons/{model}/{corpus}"
            os.makedirs(output_dir, exist_ok=True)
            bleu_output_file = os.path.join(output_dir, "bleu_scores.json")
            chrf_output_file = os.path.join(output_dir, "chrf2_scores.json")
            write_json(bleu_output_file, bleu_scores)
            write_json(chrf_output_file, chrf_scores)

        


