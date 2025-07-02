import matplotlib.pyplot as plt
from matplotlib import colors

def plot_bo_results(ground_truth_df, bo_log_df, output_path, config):
    # ... (이전 코드의 plot_results 함수와 거의 동일) ...
    # 단, config 객체에서 ALPHA_BOUNDS 등을 가져오거나,
    # ground_truth_df에서 직접 범위를 추출할 수 있습니다.
    alpha = ground_truth_df['alpha'].unique()
    th_w_ratio = ground_truth_df['th_w_ratio'].unique()
    Z = ground_truth_df['max_amplitude'].values.reshape(len(th_w_ratio), len(alpha)) # 수정: reshape 순서
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    vmin = ground_truth_df['max_amplitude'].min()
    vmax = ground_truth_df['max_amplitude'].max()
    
    contour = ax.contourf(alpha, th_w_ratio, Z, levels=20, cmap='viridis', 
                          norm=colors.LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else None)
    fig.colorbar(contour, ax=ax, label='Max Wrinkle Amplitude (Ground Truth)')
    
    bo_points = bo_log_df[bo_log_df['Iteration_Step'].str.startswith('Iter_')]
    initial_points = bo_log_df[bo_log_df['Iteration_Step'] == 'Initial']

    ax.scatter(initial_points['alpha_actual'], initial_points['th_w_ratio_actual'],
               c='white', s=120, ec='black', marker='D', label='Initial Points', zorder=3)
    
    if not bo_points.empty:
        ax.plot(bo_points['alpha_actual'], bo_points['th_w_ratio_actual'],
                'r-o', markersize=8, label='MFBO Path', zorder=4)
            
    final_rec = bo_log_df[bo_log_df['Iteration_Step'].str.contains('Recommendation')]
    ax.scatter(final_rec['alpha_actual'], final_rec['th_w_ratio_actual'],
               c='cyan', s=250, ec='black', marker='*', label='BO Recommendation', zorder=5)
               
    gt_optimum_idx = ground_truth_df['max_amplitude'].idxmax() # 최대화 문제이므로 idxmax
    gt_optimum = ground_truth_df.loc[gt_optimum_idx]
    ax.scatter(gt_optimum['alpha'], gt_optimum['th_w_ratio'],
               c='magenta', s=250, ec='black', marker='P', label='Ground Truth Optimum', zorder=5)

    ax.set_xlabel('Aspect Ratio (alpha)')
    ax.set_ylabel('Width-to-Thickness Ratio (Wo/to)')
    ax.set_title(f'MFBO Path vs. Ground Truth\n({config.EXPERIMENT_NAME})')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")