import torch
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity

def mfkg_acq_f(model, train_X, train_Y, cost_utility, config, fidelity_idx):
    
    def project_func(X):
        return project_to_target_fidelity(X=X, target_fidelities={fidelity_idx: config.TARGET_FIDELITY_VALUE})

    hf_mask = (train_X[:, fidelity_idx] == config.TARGET_FIDELITY_VALUE)
    best_observed_value = train_Y[hf_mask].max() if hf_mask.any() else -torch.inf
    
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=best_observed_value,
        cost_aware_utility=cost_utility,
        project=project_func
    )