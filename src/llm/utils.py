import json, yaml

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
        json.dump(data, f)

def read_yaml(file):
    with open(file, "r") as f:
        return yaml.safe_load(f)