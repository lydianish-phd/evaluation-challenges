
import os, argparse, json

MINOR = "minor"
MAJOR = "major"
CRITICAL = "critical"

def count_error_types(errors):
    error_stats = {
        MINOR: 0,
        MAJOR: 0,
        CRITICAL: 0
    }
    for error in errors:
        for span in error["spans"]:
            error_stats[span["severity"]] += 1
    error_stats["total"] = sum(error_stats.values())
    return error_stats

def get_sentences_with_errors(errors, severity):
    sentence_ids = []
    for i, sentence in enumerate(errors):
        for span in sentence["spans"]:
            if span["severity"] == severity:
                sentence_ids.append(i)
                break
    return set(sentence_ids)

def get_correct_sentences(errors):
    sentence_ids = []
    for i, sentence in enumerate(errors):
        if len(sentence["spans"]) == 0:
            sentence_ids.append(i)
    return set(sentence_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Path to the errors file")
    args = parser.parse_args()

    with open(args.input_file) as f:
        errors = json.load(f)

    error_stats = count_error_types(errors)

    for severity in [MINOR, MAJOR, CRITICAL]:
        error_stats[f"{severity}_sents_ids"] = get_sentences_with_errors(errors, severity)
        error_stats[f"{severity}_sents"] = len(error_stats[f"{severity}_sents_ids"])
    
    error_stats["correct_sents_ids"] = get_correct_sentences(errors)
    error_stats["correct_sents"] = len(error_stats["correct_sents_ids"])

    output_file = args.input_file.replace(".errors.json", ".counts.json")  
    with open(output_file, "w") as f:
        json.dump(error_stats, f, indent=4)
