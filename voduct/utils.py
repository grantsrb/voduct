import json
import os

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

