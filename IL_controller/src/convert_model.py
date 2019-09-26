import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True)
parser.add_argument("--dest", type=str, required=True)

args = parser.parse_args()

model_state = torch.load(args.source)
new_model_state = {}

for key in model_state.keys():
    new_model_state[key[7:]] = model_state[key]

torch.save(new_model_state, args.dest)