#!/usr/bin/env python

"""
    Loads up the `model.yaml` in the current directory, and runs it.
"""

import sys
import subprocess
import ruamel.yaml

def run_step(model, step):
    exe             = model["steps"][step]["entrypoint"]
    params          = model["steps"][step]["params"]
    common_params   = model["params"]

    merged = common_params
    for (k, v) in params.items():
        merged[k] = v
        
    args = sum([["--" + key, str(val)] for (key, val) in merged.items()], [])

    print("Running `{}` with arguments:".format(exe))
    print(" ".join(args))

    subprocess.check_call([exe] + args)


if __name__ == "__main__":

    with open("model.yaml", "r") as f:
        yaml = f.read()

    model = ruamel.yaml.load(yaml)

    if len(sys.argv) == 2:
        step = sys.argv[1]
        run_step(model, step)
    else:
        for step in model["workflow"]:
            print("Executing step: {}.".format(step))

            run_step(model, step)

            print("")
