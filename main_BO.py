import os
import torch
import pandas as pd
import datetime
import argparse
import importlib

from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.optim.optimize import optimize_acqf_mixed

from BO_framework.initial_design import generate_initial_data_with_LHS
from BO_framework.models import initialize_gp_model, get_final_posterior_mean
from BO_framework.acquisition import mfkg_acq_f
from utils.plotting import plot_bo_results

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def get_problem_class(class_path_str):
    try:
        module_path, class_name = class_path_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class from path '{class_path_str}': {e}")

def run_bo_loop(config):

    print(f" --- Initializing for {config.SIMULATION_TOOL.upper()}  --- ")
    ProblemClass = get_problem_class(config.PROBLEM_CLASS_PATH)
    problem = ProblemClass(config)

    log_data = []
    log_columns = [
        "Iteration_Step", "Fidelity_BO", "depth_actual",
        "Objective_BO", "Objective_Actual", "Execution_Time_s"
    ]
    # 초기 데이터 생성
    train_X, train_Y, train_costs = generate_initial_data_with_LHS(
        n_lf=config.N_LF_INIT, n_hf=config.N_HF_INIT, problem_instance=problem, tkwargs=tkwargs
    )
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    # 초기 데이터 로깅
    for i in range(train_X.shape[0]):
        x_p, y_p, t_p = train_X[i], train_Y[i].item(), train_costs[i].item()
        des_act = problem.unnormalize(x_p[:problem.num_design_vars]).cpu().numpy().flatten()
        fid_bo = x_p[problem.fidelity_dim_idx].item()
        is_hf = abs(fid_bo - config.TARGET_FIDELITY_VALUE) < 1e-6
        obj_act = -y_p if problem.negate and is_hf else y_p
        log_data.append(["Initial", fid_bo, des_act[0], y_p, obj_act, t_p])
    
    # 비용 모델 설정
    lf_costs = train_costs[train_X[:, problem.fidelity_dim_idx] == 0.0]
    hf_costs = train_costs[train_X[:, problem.fidelity_dim_idx] == 1.0]
    cost_lf = lf_costs.mean().item() if len(lf_costs) > 0 else config.FALLBACK_COST_LF
    cost_hf = hf_costs.mean().item() if len(hf_costs) > 0 else config.FALLBACK_COST_HF
    cost_model = AffineFidelityCostModel(fidelity_weights={problem.fidelity_dim_idx: cost_hf - cost_lf}, fixed_cost=cost_lf)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    
    # MFBO 메인 루프
    print(f"\n--- Starting MFBO Loop ({config.NUM_BO_ITERATIONS} iterations) ---")
    for iteration in range(config.NUM_BO_ITERATIONS):
        mll, model = initialize_gp_model(train_X, train_Y, problem.fidelity_dim_idx)
        if model is None:
            print("Model initialization failed. Stopping.")
            break
        
        acqf = mfkg_acq_f(model, train_X, train_Y, cost_aware_utility, config, problem.fidelity_dim_idx)
        
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf, bounds=config.NORMALIZED_BOUNDS, q=config.BATCH_SIZE,
            num_restarts=config.NUM_RESTARTS, raw_samples=config.RAW_SAMPLES,
            fixed_features_list=[{problem.fidelity_dim_idx: 0.0}, {problem.fidelity_dim_idx: 1.0}]
        )
        
        # 새로운 후보 평가 및 데이터 업데이트
        new_Y, new_costs = problem(candidates)

        if new_Y.ndim == 1:
            new_Y = new_Y.unsqueeze(-1)

        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        
        # 새 데이터 로깅
        x_p, y_p, t_p = candidates[0], new_Y[0].item(), new_costs[0].item()
        des_act = problem.unnormalize(x_p[:problem.num_design_vars]).cpu().numpy().flatten()
        fid_bo = x_p[problem.fidelity_dim_idx].item()
        is_hf = abs(fid_bo - config.TARGET_FIDELITY_VALUE) < 1e-6
        obj_act = -y_p if problem.negate and is_hf else y_p
        log_data.append([f"Iter_{iteration+1}", fid_bo, des_act[0], y_p, obj_act, t_p])
        
        print(f"  Iteration {iteration+1}: Selected fid={fid_bo:.1f}, depth={des_act[0]:.3f}, Result(BO)={y_p:.4e}")

    # 최종 추천 및 결과 저장
    print("\n--- Final Recommendation ---")
    recommended_x_full, final_Y, final_cost = get_final_posterior_mean(train_X, train_Y, problem, config)
    
    # 추천 결과 로깅
    x_p, y_p, t_p = recommended_x_full[0], final_Y[0].item(), final_cost[0].item()
    des_act = problem.unnormalize(x_p[:problem.num_design_vars]).cpu().numpy().flatten()
    fid_bo = x_p[problem.fidelity_dim_idx].item()
    obj_act = -y_p if problem.negate else y_p
    log_data.append(["Recommendation", fid_bo, des_act[0], y_p, obj_act, t_p])
    print(f"Recommended: depth={des_act[0]:.3f}, Final Actual Objective={obj_act:.4e}")

    # 로그 파일 저장
    log_df = pd.DataFrame(log_data, columns=log_columns)
    log_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.EXPERIMENT_NAME}_log.csv"
    log_df.to_csv(log_filename, index=False)
    print(f"\nLog file saved to {log_filename}")

    # Ground Truth와 비교 분석
    if os.path.exists(config.GROUND_TRUTH_FILE):
        print("\n--- Performing Analysis with Ground Truth ---")
        gt_df = pd.read_csv(config.GROUND_TRUTH_FILE)
        plot_filename = log_filename.replace(".csv", "_analysis_plot.png")
        plot_bo_results(gt_df, log_df, plot_filename, config)
    else:
        print(f"\nWarning: Ground truth file '{config.GROUND_TRUTH_FILE}' not found. Skipping analysis plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Fidelity Bayesian Optimization.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.default",
        help="Configuration file to use (e.g., 'config.experiment_01')."
    )
    args = parser.parse_args()

    try:
        config_module = importlib.import_module(args.config)
        print(f"Successfully loaded configuration: {args.config}")
    except ImportError:
        print(f"Error: Could not import a configuration file named '{args.config}'. Please check the path.")
        exit()

    problem_class_ref = None
    try:
        ProblemClass = get_problem_class(config_module.PROBLEM_CLASS_PATH)
        problem_class_ref = ProblemClass

        run_bo_loop(config_module)
    
    except Exception as e:
        print(f"An error occurred during the BO loop: {e}")
        import traceback
        traceback.print_exc()

    #finally:
    #    if problem_class_ref and hasattr(problem_class_ref, 'cleanup'):
    #        problem_class_ref.cleanup()
    print("\nOptimization finished.")