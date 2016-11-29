#!/usr/bin/env python

"""
    Loads up the `model.yaml` in the current directory, and runs it.
"""

import sys
import subprocess
import ruamel.yaml
import copy

def run_step(model, step, remaining_args):
    exe             = model["steps"][step]["entrypoint"]
    params          = model["steps"][step]["params"]
    common_params   = model["common_params"]

    merged = copy.deepcopy(common_params)
    for (k, v) in params.items():
        merged[k] = v
    
    for i, param_name in enumerate(remaining_args):
        if param_name.startswith("--"):
            merged[param_name[2:]] = remaining_args[i+1]

    args = sum([["--" + key, str(val)] for (key, val) in merged.items()], [])

    print("Running `{}` with arguments:".format(exe))
    print(" ".join(args))

    subprocess.check_call([exe] + args)


if __name__ == "__main__":

    with open("model.yaml", "r", encoding="utf-8") as f:
        yaml = f.read()

    model = ruamel.yaml.load(yaml)

    if len(sys.argv) == 2:
        step = sys.argv[1]
        run_step(model, step, sys.argv[2:])
    else:
        for step in model["workflow"]:
            print("Executing step: {}.".format(step))

            run_step(model, step, sys.argv[1:])

            print("")
