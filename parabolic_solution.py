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

def evaluate_point(problem, depth, config):
    depth_norm = normalize(depth, config.DEPTH_BOUNDS)
    X_norm = torch.tensor([[depth_norm, 0.0]], **tkwargs)
    objective_tensor, cost_tensor = problem(X_norm)
    objective_item = objective_tensor.item()
    cost_item = cost_tensor.item()

    if np.isnan(objective_item) or abs(objective_item - problem.nan_penalty) < 1e-6:
        return {
            "depth_actual": depth,
            "objective_actual": np.nan,
            "cost_s": cost_item if not np.isnan(cost_item) else np.nan,
            "status": "Failed"
        }

    else:
        actual_objective = -objective_item if problem.negate else objective_item
        return {
            "depth_actual": depth,
            "objective_actual": actual_objective,
            "cost_s": cost_item,
            "status": "Success"
        }

def visualize_grid(points_df, output_filename="ground_truth_plot_1d.png"):
    if points_df.empty:
        print("DataFrame is empty, skipping visualization.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    success_points = points_df.dropna(subset=['objective_actual'])
    failure_points = points_df[points_df['objective_actual'].isna()]

    if not success_points.empty:
        success_points = success_points.sort_values(by='depth_actual')
        ax.plot(success_points['depth_actual'], success_points['objective_actual'],
                marker='o', markersize=5, linestyle='-', color='blue', label='Success')

    if not failure_points.empty:
        y_val_failure = ax.get_ylim()[0] if not success_points.empty else 0
        ax.scatter(failure_points['depth_actual'], [y_val_failure] * len(failure_points),
                   marker='x', color='red', label='Failure', s=100)

    ax.set_xlabel('parabolic depth')
    ax.set_ylabel('Objective')
    ax.set_title(f'Ground Truth Solution Space')
    ax.grid(True)
    plt.savefig(output_filename)
    #print(f"\nGround truth analysis plot saved to {output_filename}")
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
        #print(f"Successfully loaded configuration: {args.config}")
    except ImportError as e:
        print(f"Error loading configuration '{args.config}': {e}")
        exit()

    ProblemClass = get_problem_class(config.PROBLEM_CLASS_PATH)
    depth_vals = np.linspace(config.DEPTH_BOUNDS[0], config.DEPTH_BOUNDS[1], config.GRID_RESOLUTION)
    all_points = list(depth_vals)

    success_points, failure_points, file_buffer = [], [], []
    CHUNK_SIZE_RESTART = 10
    CHUNK_SIZE_SAVE = 10
    
    output_file_path = os.path.join(config.PROJECT_ROOT, config.GROUND_TRUTH_FILE_NAME)
    write_header = not os.path.exists(output_file_path)
    total_points = len(all_points)
    start_time = time.time()

    columns_to_save = ["depth_actual", "objective_actual", "cost_s"]
    
   # print(f"--- Starting Ground Truth Generation ({total_points} points) ---")
    #print(f"Results will be saved to: {output_file_path}")
    #print(f"Resources will restart every {CHUNK_SIZE_RESTART} evaluations.")

    with open(output_file_path, 'a', newline='') as f:
        for i in range(0, total_points, CHUNK_SIZE_RESTART):
            problem = ProblemClass(negate=False, config=config)
            chunk_points = all_points[i : i + CHUNK_SIZE_RESTART]
            
            for point_idx, depth in enumerate(chunk_points):
                global_point_num = i + point_idx + 1
                    
                result = evaluate_point(problem, depth, config)
                data_to_save = {k: result[k] for k in columns_to_save}
                file_buffer.append(data_to_save)
                
                (success_points if result["status"] == "Success" else failure_points).append(result)
                
                amp_str = f"{result['objective_actual']:.4e}" if result['status'] == "Success" else "N/A"
                print(f"  [{global_point_num}/{total_points}] depth={depth:.3f} -> obj={amp_str} | {result['status']}")

                if len(file_buffer) >= CHUNK_SIZE_SAVE:
                    pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False, columns=columns_to_save)
                    file_buffer, write_header = [], False
            
        if file_buffer:
            pd.DataFrame(file_buffer).to_csv(f, header=write_header, index=False, columns=columns_to_save)

    plot_df = pd.DataFrame(success_points + failure_points)
    plot_filename = output_file_path.replace(".csv", "_plot.png")
    visualize_grid(plot_df, plot_filename)
    
    total_time = time.time() - start_time
    print(f"\n--- Ground Truth generation complete ---")
    print(f"Total time: {total_time:.2f} seconds")