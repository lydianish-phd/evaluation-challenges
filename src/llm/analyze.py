import os, argparse, json, yaml
from sacrebleu.metrics import BLEU

from evaluate import (
    LLAMA,
    GEMMA,
    NLLB,
    CORPORA_CONFIG,
    CRITICAL,
    get_sentences_with_errors,
)

from utils import read_file, read_json, read_yaml

ROCSMT_LINE_NUMBERS = [9,11,18,36,43,51,56,80,91,97,102,120,153,160,164,180,220,248,264,283,319,347,369,416,478,536,574,593,624,633,655,666,750,780,805,807,817,838,904,906,907,922,930,951,976,980,996,1015,1034,1047,1134,1153,1193,1205,1207,1230,1275,1311,1436,1462,1468,1482,1537,1548,1593,1596,1643,1648,1665,1739,1768,1780,1803,1922]
FOOTWEETS_LINE_NUMBERS = [2,3,7,18,21,22,24,26,30,35,177,195,337,374,414,422,525,536,573,610,637,707,718,914,1011,1093,1097,1350,1566,1567,1568,1569,1571,1612,1817,1992,2148,2420,2444,2525,2526,2544,2548,2590,2669,2670,2692,2813,2853,2885,2953,3354,3490,3566,3799,3910,3933,3986]
MMTC_LINE_NUMBERS = [1, 6, 8, 82, 96, 131, 188, 289, 301, 332, 441, 457, 610, 621, 667, 792, 801, 891, 976, 1008, 1035, 1179, 1262, 1465, 1522, 1627, 1643, 1667, 1712, 1746, 1763, 1774, 1782, 1862, 1898, 1935, 1963, 1999] 
PFSMB_LINE_NUMBERS = [3,27,32,34,35,45,46,63,78,105,117,119,122,129,152,166,173,186,204,298,303,306,421,427,438,467,471,503,523,640,656,657,669,687,688,731,744,770]


SELECTED_EXAMPLES = {
    "rocsmt": [k - 1 for k in ROCSMT_LINE_NUMBERS],
    "footweets": [k - 1 for k in FOOTWEETS_LINE_NUMBERS],
    "mmtc": [k - 1 for k in MMTC_LINE_NUMBERS],
    "pfsmb": [k - 1 for k in PFSMB_LINE_NUMBERS],
}

bleu_metric = BLEU(effective_order=True)

def get_outputs(line_ids, src, ref, sys, errors, comet_scores):
    output = ""
    for i in line_ids:
        output += f"Line {i+1}\n"
        output += f"SRC: {src[i]}\n"
        output += f"REF: {ref[i]}\n"
        for guideline in sys:
            output += "------------------------------\n"
            output += f"SYS ({guideline}): {sys[guideline][i]}\n"
            output += f"ERRORS ({guideline}): {errors[guideline][i]}\n" if errors[guideline] is not None else ""
            output += f"BLEU ({guideline}): {bleu_metric.sentence_score(sys[guideline][i], [ref[i]]).score}\n"
            output += f"COMET ({guideline}): {comet_scores[guideline][i]}\n"
        output += "\n"
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+", default=["rocsmt", "footweets", "mmtc", "pfsmb"])
    parser.add_argument("-m", "--models", type=str, nargs="+", default=[LLAMA, GEMMA])
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    args = parser.parse_args()

    config = read_yaml(args.corpora_config)

    for model in args.models:
        for corpus in args.corpora:
            print(f"Analysis for {model} on {corpus}...")
            src_file = os.path.expandvars(config[corpus]["src_file_path"])
            ref_file = os.path.expandvars(config[corpus]["ref_file_path"])
            src_file_name = os.path.basename(src_file)
            
            src = read_file(src_file)
            ref = read_file(ref_file)

            sys = {}
            errors = {}
            comet_scores = {}
            for guideline in ["baseline", "default", "rocsmt", "footweets", "mmtc", "pfsmb"]:
                sys_model = NLLB if (guideline == "baseline") else model
                guideline_ext = "out" if (guideline == "baseline") else  f"{guideline}.out"
                sys_file = f"{args.input_dir}/outputs/{sys_model}/{corpus}/{src_file_name}.{guideline_ext}"
                error_file = f"{args.input_dir}/outputs/{sys_model}/{corpus}/{src_file_name}.{guideline_ext}.errors.json"
                comet_file  = f"{args.input_dir}/outputs/{sys_model}/{corpus}/{src_file_name}.{guideline_ext}.comet.json"
                sys[guideline] = read_file(sys_file)
                errors[guideline] = read_json(error_file) if os.path.exists(error_file) else None
                comet_scores[guideline] = read_json(comet_file)

            output_dir = f"{args.input_dir}/analysis/{model}/{corpus}"
            os.makedirs(output_dir, exist_ok=True)

            print(f" - Collecting selected examples")
            with open(f"{output_dir}/selected_examples.txt", "w") as f:
                f.write(get_outputs(SELECTED_EXAMPLES[corpus], src, ref, sys, errors))
            
            critical_errors = set()
            for guideline in errors:
                if errors[guideline] is not None:
                    critical_errors = critical_errors.union(get_sentences_with_errors(errors[guideline], CRITICAL))
            
            print(f" - Collecting critical errors")
            with open(f"{output_dir}/critical_errors.txt", "w") as f:
                f.write(get_outputs(critical_errors, src, ref, sys, errors))
                



