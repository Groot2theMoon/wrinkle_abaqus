import numpy as np
import pandas as pd
import os
import torch
import time
import matplotlib.pyplot as plt
import importlib
from abaqus_interface.problem_definition import AbaqusWrinkleFunction

try:
    config = importlib.import_module("config.default")
    print("Loaded configuration from config.default")
except ImportError:
    print("Error: Could not load config/default.py. Please check the file.")
    exit()

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def normalize(value, bounds):
    return (value - bounds[0]) / (bounds[1] - bounds[0])

def evaluate_point(problem, alpha, th_w_ratio, config_obj):
    alpha_norm = normalize(alpha, config_obj.ALPHA_BOUNDS)
    th_w_norm = normalize(th_w_ratio, config_obj.TH_W_RATIO_BOUNDS)
    X_norm = torch.tensor([[alpha_norm, th_w_norm, 1.0]], **tkwargs)

    max_amplitude_tensor, cost_tensor = problem(X_norm)
    
    max_amplitude_item = max_amplitude_tensor.item()
    cost_item = cost_tensor.item()

    if np.isnan(max_amplitude_item) or abs(max_amplitude_item - problem.nan_penalty) < 1e-6:
        return {
            "alpha": alpha, "th_w_ratio": th_w_ratio,
            "max_amplitude": np.nan,
            "cost_s": cost_item if not np.isnan(cost_item) else np.nan,
            "status": "Failed: ABAQUS execution error"
        }
    else:
        return {
            "alpha": alpha, "th_w_ratio": th_w_ratio,
            "max_amplitude": max_amplitude_item,
            "cost_s": cost_item,
            "status": "Success"
        }

def visualize_grid(success_points, failure_points, output_filename="ground_truth_status.png"):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    if success_points:
        s_alphas = [p["alpha"] for p in success_points]
        s_ratios = [p["th_w_ratio"] for p in success_points]
        ax.scatter(s_alphas, s_ratios, c='green', marker='o', s=80, 
                   label=f'Success ({len(success_points)})', alpha=0.8)

    if failure_points:
        f_alphas = [p["alpha"] for p in failure_points]
        f_ratios = [p["th_w_ratio"] for p in failure_points]
        ax.scatter(f_alphas, f_ratios, c='red', marker='x', s=100, 
                   label=f'Failure ({len(failure_points)})', alpha=1.0)

    ax.set_xlabel('alpha')
    ax.set_ylabel('Wo/t')
    ax.set_title('Ground Truth Grid')
    
    all_ratios = [p["th_w_ratio"] for p in success_points + failure_points]
    if all_ratios and (max(all_ratios) / min(all_ratios)) > 100:
        ax.set_yscale('log')

    plt.savefig(output_filename)
    print(f"\nGrid status plot saved to {output_filename}")
    plt.close(fig)

if __name__ == "__main__":

    problem = AbaqusWrinkleFunction(negate=False, config=config)

    alpha_vals = np.linspace(config.ALPHA_BOUNDS[0], config.ALPHA_BOUNDS[1], config.GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(config.TH_W_RATIO_BOUNDS[0], config.TH_W_RATIO_BOUNDS[1], config.GRID_RESOLUTION)
    
    success_points, failure_points, file_buffer = [], [], []
    CHUNK_SIZE = 10
    
    output_file_path = os.path.join(config.PROJECT_ROOT, "ground_truth.csv")
    write_header = not os.path.exists(output_file_path)
    total_points = config.GRID_RESOLUTION**2
    start_time = time.time()
    
    print(f"--- Starting Ground Truth Generation ({total_points} points) ---")
    print(f"Results will be saved to: {output_file_path}")

    with open(output_file_path, 'a', newline='') as f:
        for i, alpha in enumerate(alpha_vals):
            for j, th_w_ratio in enumerate(th_w_ratio_vals):
                
                result = evaluate_point(problem, alpha, th_w_ratio, config)
                
                # 파일 저장용 데이터만 버퍼에 추가
                file_buffer.append({k: result[k] for k in ["alpha", "th_w_ratio", "max_amplitude", "cost_s"]})
                
                # 시각화용 데이터 분리
                (success_points if result["status"] == "Success" else failure_points).append(result)
                
                # 진행 상황 출력
                point_num = i * config.GRID_RESOLUTION + j + 1
                amp_str = f"{result['max_amplitude']:.4e}" if result['status'] == "Success" else "N/A"
                print(f"  [{point_num}/{total_points}] α={alpha:.3f}, Wo/to={th_w_ratio:.1f} -> amp={amp_str} | {result['status']}")
                
                # 청크 저장 로직
                if len(file_buffer) >= CHUNK_SIZE:
                    pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False)
                    file_buffer, write_header = [], False
        
        # 잔여 데이터 처리
        if file_buffer:
            pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False)

    # 시각화 실행
    visualize_grid(success_points, failure_points)
    
    total_time = time.time() - start_time
    print(f"\n--- Ground Truth generation complete ---")
    print(f"Total time: {total_time:.2f}s")