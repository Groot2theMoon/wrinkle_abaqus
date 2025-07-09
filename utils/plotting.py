# utils/plotting.py

import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

def plot_bo_results(ground_truth_df, bo_log_df, output_path, config, problem):
    """
    Plots the results of the Bayesian Optimization, adapting the plot type
    based on the number of design variables. This version dynamically detects
    column names to avoid KeyErrors.
    """
    num_dims = problem.num_design_vars

    # --- 동적 컬럼 이름 감지 ---
    # BO 로그의 컬럼 구조: ["Iter_Step", "Fid_BO", DV1, DV2, ..., "Obj_BO", "Obj_Actual", "Time"]
    bo_dv_names = list(bo_log_df.columns[2 : 2 + num_dims])
    
    # Ground Truth 로그의 컬럼 구조: [DV1, DV2, ..., Objective, Cost]
    gt_dv_names = list(ground_truth_df.columns[:num_dims])
    gt_obj_name = ground_truth_df.columns[num_dims]
    # --- 감지 끝 ---

    if num_dims == 1:
        plot_1d_results(ground_truth_df, bo_log_df, output_path, config, problem,
                        bo_dv_names[0], gt_dv_names[0], gt_obj_name)
    elif num_dims == 2:
        plot_2d_results(ground_truth_df, bo_log_df, output_path, config, problem,
                        bo_dv_names, gt_dv_names, gt_obj_name)
    else: # 3D or more
        plot_convergence(bo_log_df, output_path, config)


def plot_1d_results(gt_df, log_df, output_path, config, problem,
                    bo_dv_name, gt_dv_name, gt_obj_name):
    """Plots results for a 1D optimization problem."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ground Truth 라인 플롯
    ax.plot(gt_df[gt_dv_name], gt_df[gt_obj_name], 'k-', label='Ground Truth', alpha=0.7)
    
    # BO가 탐색한 점들
    initial_points = log_df[log_df['Iteration_Step'] == 'Initial']
    bo_points = log_df[log_df['Iteration_Step'].str.startswith('Iter_')]
    
    ax.scatter(initial_points[bo_dv_name], initial_points['Objective_Actual'], 
               c='gray', marker='D', s=100, ec='black', label='Initial Points', zorder=3)
    ax.scatter(bo_points[bo_dv_name], bo_points['Objective_Actual'], 
               c=bo_points['Fidelity_BO'], cmap='coolwarm', marker='o', s=80, ec='black', label='BO evaluations (color=fidelity)', zorder=4)
    
    # 최종 추천 및 Ground Truth 최적점
    final_rec = log_df[log_df['Iteration_Step'] == 'Recommendation']
    ax.scatter(final_rec[bo_dv_name], final_rec['Objective_Actual'],
               c='cyan', s=250, ec='black', marker='*', label='BO Recommendation', zorder=5)

    gt_optimum_idx = gt_df[gt_obj_name].idxmin() if problem.negate else gt_df[gt_obj_name].idxmax()
    gt_optimum = gt_df.loc[gt_optimum_idx]
    ax.scatter(gt_optimum[gt_dv_name], gt_optimum[gt_obj_name],
               c='magenta', s=250, ec='black', marker='P', label='Ground Truth Optimum', zorder=5)

    ax.set_xlabel(f'Design Variable ({gt_dv_name})')
    ax.set_ylabel('Objective Value')
    ax.set_title(f'1D MFBO Result vs. Ground Truth\n({config.EXPERIMENT_NAME})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")
    plt.close(fig)


def plot_2d_results(gt_df, log_df, output_path, config, problem,
                    bo_dv_names, gt_dv_names, gt_obj_name):
    """Plots results for a 2D optimization problem."""
    fig, ax = plt.subplots(figsize=(12, 9))

    bo_dv1, bo_dv2 = bo_dv_names
    gt_dv1, gt_dv2 = gt_dv_names

    # Ground Truth 컨투어 플롯
    pivot_table = gt_df.pivot_table(index=gt_dv2, columns=gt_dv1, values=gt_obj_name)
    X, Y, Z = pivot_table.columns.values, pivot_table.index.values, pivot_table.values
    
    vmin, vmax = gt_df[gt_obj_name].min(), gt_df[gt_obj_name].max()
    is_log_scale = vmin > 0 and (vmax / vmin) > 100

    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', 
                          norm=colors.LogNorm(vmin=vmin, vmax=vmax) if is_log_scale else None)
    fig.colorbar(contour, ax=ax, label=f'Objective ({gt_obj_name})')
    
    # BO 경로 플롯
    initial_points = log_df[log_df['Iteration_Step'] == 'Initial']
    bo_points = log_df[log_df['Iteration_Step'].str.startswith('Iter_')]
    final_rec = log_df[log_df['Iteration_Step'] == 'Recommendation']
    
    ax.scatter(initial_points[bo_dv1], initial_points[bo_dv2], c='white', s=120, ec='black', marker='D', label='Initial Points', zorder=3)
    if not bo_points.empty:
        ax.plot(bo_points[bo_dv1], bo_points[bo_dv2], 'r-o', markersize=8, label='MFBO Path', zorder=4)

    ax.scatter(final_rec[bo_dv1], final_rec[bo_dv2], c='cyan', s=250, ec='black', marker='*', label='BO Recommendation', zorder=5)

    # Ground Truth 최적점
    gt_optimum_idx = gt_df[gt_obj_name].idxmin() if problem.negate else gt_df[gt_obj_name].idxmax()
    gt_optimum = gt_df.loc[gt_optimum_idx]
    ax.scatter(gt_optimum[gt_dv1], gt_optimum[gt_dv2], c='magenta', s=250, ec='black', marker='P', label='Ground Truth Optimum', zorder=5)

    ax.set_xlabel(f'Design Variable ({gt_dv1})')
    ax.set_ylabel(f'Design Variable ({gt_dv2})')
    ax.set_title(f'2D MFBO Path vs. Ground Truth\n({config.EXPERIMENT_NAME})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")
    plt.close(fig)


def plot_convergence(log_df, output_path, config):
    """Plots the convergence of the objective function over iterations for high-dimensional problems."""
    fig, ax = plt.subplots(figsize=(12, 8))

    hf_evals = log_df[log_df['Fidelity_BO'] == config.TARGET_FIDELITY_VALUE].copy()
    hf_evals = hf_evals[hf_evals['Iteration_Step'] != 'Recommendation']
    
    if hf_evals.empty:
        print("No high-fidelity evaluations to plot for convergence. Skipping.")
        plt.close(fig)
        return

    hf_evals['Iter_Num'] = hf_evals['Iteration_Step'].apply(lambda x: 0 if x == 'Initial' else int(x.split('_')[1]))
    hf_evals = hf_evals.sort_values('Iter_Num')
    
    hf_evals['Best_so_far'] = hf_evals['Objective_Actual'].cummin()
    
    ax.plot(hf_evals['Iter_Num'], hf_evals['Best_so_far'], 'r-o', label='Best Objective Found (HF)')
    ax.scatter(hf_evals['Iter_Num'], hf_evals['Objective_Actual'], c='blue', marker='x', label='HF Evaluations')

    ax.set_xlabel('BO Iteration Number (Initial Points as 0)')
    ax.set_ylabel('Best Objective Value Found So Far')
    ax.set_title(f'Convergence Plot\n({config.EXPERIMENT_NAME})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")
    plt.close(fig)