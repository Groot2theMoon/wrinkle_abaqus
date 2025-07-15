import torch
import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf

def initialize_gp_model(train_X, train_Y, fidelity_dim_idx):

    valid_idx = ~torch.isinf(train_Y.squeeze()) & ~torch.isnan(train_Y.squeeze())
    if valid_idx.sum() < 2:
        print("Warning: Not enough valid data points to initialize model.")
        return None, None

    model = SingleTaskMultiFidelityGP(
        train_X=train_X[valid_idx], 
        train_Y=train_Y[valid_idx],
        data_fidelities={fidelity_dim_idx: 0},
        outcome_transform=Standardize(m=train_Y[valid_idx].shape[-1]), 
    )
    
    # --- 모델이 과도하게 확신하는 것을 방지하기 위한 하이퍼파라미터 제약 추가 ---
    # 이 부분이 "Unable to find non-zero acquisition function" 문제를 해결하는 데 매우 중요합니다.
    try:
        model.covar_module.base_kernel.lengthscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
        model.covar_module.outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
        
        likelihood = model.likelihood
        if hasattr(likelihood, "noise_covar"):
             likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-6))
    except AttributeError:
        # 모델 구조가 다를 경우를 대비한 예외 처리
        print("Warning: Could not set priors on model hyperparameters.")
        pass
    # --- 제약 추가 끝 ---

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    try:
        fit_gpytorch_mll(mll)
    except Exception as e:
        print(f"Warning: GP model fitting failed: {e}. Returning None.")
        return None, None
        
    return mll, model

def get_final_posterior_mean(train_X, train_Y, problem, config):
    _, model = initialize_gp_model(train_X, train_Y, problem.fidelity_dim_idx)
    
    if model is None:
        print("Final model could not be initialized. Cannot provide recommendation.")

        hf_mask = (train_X[:, problem.fidelity_dim_idx] == config.TARGET_FIDELITY_VALUE)
        if hf_mask.any():
            best_idx = train_Y[hf_mask].argmax()
            recommended_x_full = train_X[hf_mask][best_idx].unsqueeze(0)
            print("Recommending the best observed high-fidelity point as a fallback.")
            final_Y = train_Y[hf_mask][best_idx].unsqueeze(0)
            
            final_cost = torch.tensor([[0.0]], dtype=train_X.dtype, device=train_X.device) 
            return recommended_x_full, final_Y, final_cost
        else:
            raise RuntimeError("No valid model and no high-fidelity points to recommend.")

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
    
    #print("Evaluating final recommended point at High Fidelity...")
    final_Y, final_cost = problem(recommended_x_full)
    return recommended_x_full, final_Y, final_cost