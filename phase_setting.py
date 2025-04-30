import torch
import numpy as np

def get_phase_config(args):

    num_phases = 2 ** args.bits
    # print(f"main.py/get_phases_discrete || num_phases_global: {num_phases}")

    phases_discrete = torch.linspace(
        start=0.0,
        end=2 * np.pi * (num_phases - 1) / num_phases,
        steps=num_phases,
        dtype=torch.float32
    ).to(args.device)
    # print(f"main.py/get_phases_discrete || phases_discrete_global: {phases_discrete}")

    return num_phases, phases_discrete.cpu().numpy()