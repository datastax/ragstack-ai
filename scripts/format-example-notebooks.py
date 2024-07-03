#!/usr/bin/env python

import json
import os


def main():
    nb_make_final_object = {
        "post_cell_execute": [
            "from conftest import before_notebook",
            "before_notebook()",
        ]
    }

    nb_path = os.path.join("examples", "notebooks")
    for root, _, files in os.walk(nb_path):
        for file in files:
            file = os.path.join(root, file)
            if ".ipynb_checkpoints" not in file and file.endswith(".ipynb"):
                print("Formatting file: ", file)
                with open(file) as f:
                    contents = f.read()
                    as_json = json.loads(contents)
                    cells = as_json["cells"]
                    found = False
                    for cell in cells:
                        if cell["cell_type"] == "code":
                            metadata = cell["metadata"] or {}
                            metadata["nbmake"] = nb_make_final_object
                            cell["metadata"] = metadata
                            found = True
                            break
                    if not found:
                        raise ValueError("No code cells found in file: ", file)
                with open(file, "w") as f:
                    f.write(json.dumps(as_json, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
