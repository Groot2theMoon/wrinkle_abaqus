import torch
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf

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

def find_best_candidate_mfei(
    model, train_X, train_Y, cost_model, config, problem
):
    """
    Finds the best candidate point for the next evaluation by manually optimizing
    a cost-weighted Expected Improvement across different fidelities.
    This approach avoids using the potentially unstable `optimize_acqf_mixed`.

    Args:
        model: The Gaussian Process model.
        train_X: The training data inputs.
        train_Y: The training data outputs.
        cost_model: The cost model for different fidelities.
        config: The configuration object.
        problem: The problem instance, containing dimensionality info.

    Returns:
        A (q, d) tensor representing the best candidate(s) for the next evaluation.
    """
    
    # 1. 기본 Expected Improvement 획득 함수 생성
    hf_mask = (train_X[:, problem.fidelity_dim_idx] == config.TARGET_FIDELITY_VALUE)
    best_f = train_Y[hf_mask].max() if hf_mask.any() else torch.tensor(float("-inf"), dtype=train_Y.dtype, device=train_Y.device)
    
    acqf_ei = qExpectedImprovement(model=model, best_f=best_f)

    # 2. 각 충실도에 대한 후보와 가성비(value)를 저장할 리스트
    candidates_list = []
    values_list = []
    
    fixed_features_options = [{problem.fidelity_dim_idx: 0.0}, {problem.fidelity_dim_idx: 1.0}]

    for fixed_features in fixed_features_options:
        fidelity_value = list(fixed_features.values())[0]
        
        # 2-1. 특정 충실도를 고정하는 획득 함수 래퍼 생성
        acqf_fixed_fidelity = FixedFeatureAcquisitionFunction(
            acq_function=acqf_ei,
            d=problem.dim,
            columns=[problem.fidelity_dim_idx],
            values=[fidelity_value]
        )

        # 2-2. `optimize_acqf`로 설계 변수 공간에 대해 최적화
        candidate_design, acq_value = optimize_acqf(
            acq_function=acqf_fixed_fidelity,
            bounds=config.NORMALIZED_BOUNDS[:, :problem.num_design_vars],
            q=config.BATCH_SIZE,
            num_restarts=config.NUM_RESTARTS,
            raw_samples=config.RAW_SAMPLES,
        )

        # 2-3. 전체 후보점(설계 변수 + 충실도) 복원
        candidate_full = torch.cat(
            [candidate_design, torch.tensor([[fidelity_value]], dtype=train_X.dtype, device=train_X.device)], 
            dim=-1
        )

        # 2-4. 비용 계산 및 가성비(acq_value / cost) 계산
        cost = cost_model(candidate_full).squeeze()
        cost_weighted_value = acq_value / (cost + 1e-9)

        candidates_list.append(candidate_full)
        values_list.append(cost_weighted_value)

    # 3. LF와 HF의 가성비를 비교하여 최종 후보 선택
    best_idx = torch.stack(values_list).argmax()
    best_candidate = candidates_list[best_idx]
    
    # 최종 후보점과 그 때의 (비용 가중되지 않은) 획득 함수 값을 반환
    return best_candidate, values_list[best_idx]