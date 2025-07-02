# bo_framework/initial_design.py
import torch
import numpy as np
from scipy.stats.qmc import LatinHypercube
from sklearn.utils import shuffle

def generate_initial_data_with_LHS(n_lf, n_hf, problem_instance, tkwargs):
    print(f"\n--- Generating {n_lf} LF and {n_hf} HF initial data points using LHS ---")
    
    sampler = LatinHypercube(d=problem_instance.num_design_vars)
    X_norm_lf = torch.from_numpy(sampler.random(n=n_lf)).to(**tkwargs)
    
    indices = shuffle(np.arange(n_lf))
    X_norm_hf = X_norm_lf[indices[:n_hf]]
    
    x_lf = torch.cat([X_norm_lf, torch.full((n_lf, 1), 0.0, **tkwargs)], dim=1)
    x_hf = torch.cat([X_norm_hf, torch.full((n_hf, 1), 1.0, **tkwargs)], dim=1)
    
    x_init = torch.cat([x_lf, x_hf], dim=0)
    
    print("Evaluating initial points... (this may take a while)")
    y_init, c_init = problem_instance(x_init)
    
    return x_init, y_init, c_init