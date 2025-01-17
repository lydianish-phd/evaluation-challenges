import argparse, json

MINOR = "minor"
MAJOR = "major"
CRITICAL = "critical"

def count_error_types(errors):
    error_types = {
        MINOR: 0,
        MAJOR: 0,
        CRITICAL: 0
    }
    new_errors = []
    for error in errors:
        # if there is a key named span rename it to spans
        if "span" in error:
            error["spans"] = error.pop("span")
            new_errors.append(error)
        for span in error["spans"]:
            error_types[span["severity"]] += 1
    error_types["total"] = sum(error_types.values())
    return new_errors, error_types

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        errors = json.load(f)

    new_errors, error_types = count_error_types(errors)

    if len(new_errors) > 0:
        with open(args.input_file, "w") as f:
            json.dump(new_errors, f)

    output_file = args.input_file.replace("errors.json", ".counts.json")

    with open(output_file, "w") as f:
        json.dump(error_types, f)