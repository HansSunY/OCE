import argparse
import torch
import torch_fidelity

parser = argparse.ArgumentParser()
parser.add_argument("--input1", type=str, required=True,
                    help="Path to generated images")
parser.add_argument("--input2", type=str, required=True,
                    help="Path to reference/original images")

args = parser.parse_args()

FIDELITY = torch_fidelity.calculate_metrics(
    input1=args.input1,
    input2=args.input2,
    cuda=True,
    fid=True,
    verbose=False,
)

print(FIDELITY)