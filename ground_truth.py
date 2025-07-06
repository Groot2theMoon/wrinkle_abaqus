import numpy as np
import pandas as pd
import os
import torch
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import importlib
import argparse

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

def normalize(value, bounds):
    return (value - bounds[0]) / (bounds[1] - bounds[0])

def get_problem_class(class_path_str):
    try:
        module_path, class_name = class_path_str.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class from path '{class_path_str}': {e}")

def evaluate_point(problem, alpha, th_w_ratio, config):
    alpha_norm = normalize(alpha, config.ALPHA_BOUNDS)
    th_w_norm = normalize(th_w_ratio, config.TH_W_RATIO_BOUNDS)
    X_norm = torch.tensor([[alpha_norm, th_w_norm, 1.0]], **tkwargs)

    objective_tensor, cost_tensor = problem(X_norm)
    
    objective_item = objective_tensor.item()
    cost_item = cost_tensor.item()

    if np.isnan(objective_item) or abs(objective_item - problem.nan_penalty) < 1e-6:
        return {
            "alpha": alpha, "th_w_ratio": th_w_ratio,
            "objective": np.nan, 
            "cost_s": cost_item if not np.isnan(cost_item) else np.nan,
            "status": "Failed"
        }
    else:
        return {
            "alpha": alpha, "th_w_ratio": th_w_ratio,
            "objective": objective_item,
            "cost_s": cost_item,
            "status": "Success"
        }

def visualize_grid(points_df, fidelity, output_filename="ground_truth_status.png"):
    if points_df.empty:
        print("DataFrame is empty, skipping visualization.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    success_points = points_df.dropna(subset=['objective'])
    failure_points = points_df[points_df['objective'].isna()]

    if not success_points.empty:
        try:
            pivot_df = success_points.pivot(index='th_w_ratio', columns='alpha', values='objective')
            X = pivot_df.columns.values
            Y = pivot_df.index.values
            Z = pivot_df.values

            obj_min, obj_max = success_points['objective'].min(), success_points['objective'].max()
            norm = None
            if obj_min > 0 and obj_max / obj_min > 100:
                norm = colors.LogNorm(vmin=obj_min, vmax=obj_max)

            cp = ax.contourf(X, Y, Z, levels=20, cmap='viridis', norm=norm, alpha=0.9)
            ax.contour(X, Y, Z, levels=cp.levels, colors='white', linewidths=0.5, alpha=0.7)
            fig.colorbar(cp, ax=ax, label=f'Objective Value (Fidelity {fidelity})')
        except Exception as e:
            print(f"Could not create contour plot, plotting scatter only. Error: {e}")


    if not success_points.empty:
        ax.scatter(success_points['alpha'], success_points['th_w_ratio'], 
                   c='lime', marker='o', s=50, ec='black', lw=0.5,
                   label=f'Success ({len(success_points)})', zorder=2)

    if not failure_points.empty:
        ax.scatter(failure_points['alpha'], failure_points['th_w_ratio'], 
                   c='red', marker='x', s=100,
                   label=f'Failure ({len(failure_points)})', zorder=3)

    ax.set_xlabel('Aspect Ratio (alpha)')
    ax.set_ylabel('th/W Ratio') 
    ax.set_title(f'Ground Truth Solution Space (Fidelity = {fidelity})')
    ax.legend()
    
    all_ratios = points_df['th_w_ratio'].dropna()
    if not all_ratios.empty and (all_ratios.max() / all_ratios.min()) > 100:
        ax.set_yscale('log')

    plt.savefig(output_filename)
    print(f"\nGround truth analysis plot saved to {output_filename}")
    plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate ground truth solution space.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Configuration file to use (e.g., 'config.default_comsol')."
    )
    args = parser.parse_args()

    try:
        config = importlib.import_module(args.config)
        print(f"Successfully loaded configuration: {args.config}")
    except ImportError as e:
        print(f"Error loading configuration '{args.config}': {e}")
        exit()

    ProblemClass = get_problem_class(config.PROBLEM_CLASS_PATH)

    alpha_vals = np.linspace(config.ALPHA_BOUNDS[0], config.ALPHA_BOUNDS[1], config.GRID_RESOLUTION)
    th_w_ratio_vals = np.linspace(config.TH_W_RATIO_BOUNDS[0], config.TH_W_RATIO_BOUNDS[1], config.GRID_RESOLUTION)
    
    all_points = []
    for alpha in alpha_vals:
        for th_w_ratio in th_w_ratio_vals:
            all_points.append((alpha, th_w_ratio))

    success_points, failure_points, file_buffer = [], [], []
    CHUNK_SIZE_RESTART = 10  # 10번 평가 후 COMSOL/MATLAB 재시작
    CHUNK_SIZE_SAVE = 10     # 10개 결과마다 파일에 저장
    
    output_file_path = os.path.join(config.PROJECT_ROOT, config.GROUND_TRUTH_FILE_NAME)
    write_header = not os.path.exists(output_file_path)
    total_points = len(all_points)
    start_time = time.time()
    
    print(f"--- Starting Ground Truth Generation ({total_points} points) ---")
    print(f"Results will be saved to: {output_file_path}")
    print(f"Resources will restart every {CHUNK_SIZE_RESTART} evaluations.")

    try:
        with open(output_file_path, 'a', newline='') as f:
            for i in range(0, total_points, CHUNK_SIZE_RESTART):

                # 첫 번째 청크가 아닐 경우 이전 리소스를 정리
                if i > 0:
                    ProblemClass.cleanup()
                
                ProblemClass.start_resources(config)
                
                problem = ProblemClass(negate=False, config=config)
                
                chunk_points = all_points[i : i + CHUNK_SIZE_RESTART]
                
                for point_idx, (alpha, th_w_ratio) in enumerate(chunk_points):
                    
                    global_point_num = i + point_idx + 1
                    
                    result = evaluate_point(problem, alpha, th_w_ratio, config)
                    
                    file_buffer.append({k: result[k] for k in ["alpha", "th_w_ratio", "objective", "cost_s"]})
                    (success_points if result["status"] == "Success" else failure_points).append(result)
                    
                    amp_str = f"{result['objective']:.4e}" if result['status'] == "Success" else "N/A"
                    print(f"  [{global_point_num}/{total_points}] α={alpha:.3f}, th/W={th_w_ratio:.6f} -> obj={amp_str} | {result['status']}")

                    if len(file_buffer) >= CHUNK_SIZE_SAVE:
                        pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False)
                        file_buffer, write_header = [], False
            
            if file_buffer:
                pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False)

    finally:
        print("\n--- Final cleanup of resources ---")
        if 'ProblemClass' in locals() and hasattr(ProblemClass, 'cleanup'):
            ProblemClass.cleanup()

    plot_df = pd.DataFrame(success_points + failure_points)
    plot_filename = output_file_path.replace(".csv", "_status.png")
    visualize_grid(plot_df, 1.0, plot_filename)
    
    total_time = time.time() - start_time
    print(f"\n--- Ground Truth generation complete ---")
    print(f"Total time: {total_time:.2f} seconds")