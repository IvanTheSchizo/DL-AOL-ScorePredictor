# src/utils.py
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed set to {seed}")

def save_plot(fig, filename, output_dir="outputs/plots"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    print(f"Plot saved to {path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)