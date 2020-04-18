
import json

import sys


original_path = "./something.json"

dic = {"aodo": 1, "dfjwe[": 2, "fkdjewg": 3}

json_object = json.dumps(dic, indent=4)

with open(original_path, "w") as fd:
    fd.write(json_object)
    # fd.write(json_object)

with open(original_path, "r") as fd:
    newdic = json.load(fd)

print(newdic)
