import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf

def initialize_gp_model(train_X, train_Y, fidelity_idx):
    valid_idx = ~torch.isinf(train_Y.squeeze()) & ~torch.isnan(train_Y.squeeze())
    if valid_idx.sum() < 2: return None, None
        
    model = SingleTaskMultiFidelityGP(
        train_X=train_X[valid_idx], train_Y=train_Y[valid_idx],
        outcome_transform=Standardize(m=train_Y[valid_idx].shape[-1]), 
        data_fidelities=[fidelity_idx]
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model

def get_final_posterior_mean(train_X, train_Y, problem, config):
    _, model = initialize_gp_model(train_X, train_Y, problem.fidelity_dim_idx)
    
    rec_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=problem.dim,
        columns=[problem.fidelity_dim_idx],
        values=[config.TARGET_FIDELITY_VALUE]
    )
    
    recommended_x_design_norm, _ = optimize_acqf(
        acq_function=rec_acqf,
        bounds=config.NORMALIZED_BOUNDS[:, :problem.num_design_vars],
        q=1, num_restarts=config.NUM_RESTARTS*2, raw_samples=config.RAW_SAMPLES*2
    )
    
    recommended_x_full = rec_acqf._construct_X_full(recommended_x_design_norm)
    
    print("Evaluating final recommended point at High Fidelity...")
    final_Y, final_cost = problem(recommended_x_full)
    return recommended_x_full, final_Y, final_cost