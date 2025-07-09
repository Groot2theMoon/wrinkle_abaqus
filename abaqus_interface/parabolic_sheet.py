import os
import torch
import subprocess
import time

class AbaqusParabolicWrinkle:
    def __init__(self, negate=True, config=None):
        if config is None:
            raise ValueError("Configuration object must be provided.")
        
        self.negate = negate # 최소화문제
        self.config = config
        
        self._bounds = [config.DEPTH_BOUNDS]
        self.num_design_vars = 1
        self.dim = self.num_design_vars + 1
        self.fidelity_dim_idx = self.num_design_vars
        
        self.nan_penalty = -1e10 # NaN 값 페널티는 최대화&최소화 둘 다 아주 작은 음수

        if not os.path.exists(config.ABAQUS_WORKING_DIR):
            os.makedirs(config.ABAQUS_WORKING_DIR)

    def unnormalize(self, X_norm_design_vars):
        if X_norm_design_vars.ndim == 1:
            X_norm_design_vars = X_norm_design_vars.unsqueeze(0)
        X_unnorm = torch.zeros_like(X_norm_design_vars)
        for i in range(self.num_design_vars):
            min_val, max_val = self._bounds[i]
            X_unnorm[..., i] = X_norm_design_vars[..., i] * (max_val - min_val) + min_val
        return X_unnorm

    def __call__(self, X_full_norm):
        
        if X_full_norm.ndim == 1: 
            X_full_norm = X_full_norm.unsqueeze(0)
        
        batch_size = X_full_norm.shape[0]
        objectives = torch.full((batch_size, 1), float('nan'), dtype=torch.double)
        costs = torch.full((batch_size, 1), float('nan'), dtype=torch.double)

        for i in range(batch_size):
            X_design_norm = X_full_norm[i, :self.num_design_vars]
            fidelity_bo = X_full_norm[i, self.fidelity_dim_idx].item()
            X_design_unnorm = self.unnormalize(X_design_norm)
            
            depth = X_design_unnorm[0, 0].item()
            
            job_name = f"job_alpha{depth:.3f}_fid{fidelity_bo:.0f}"
            result_file_path = os.path.join(self.config.ABAQUS_WORKING_DIR, f"result_{job_name}.txt")
            
            print(f"  > Evaluating: depth={depth:.3f}, fid={fidelity_bo:.1f}...")
            
            start_time = time.time()
            try:
                if os.path.exists(result_file_path):
                    os.remove(result_file_path)

                cmd = [
                    self.config.ABAQUS_EXE_PATH, "cae", f"noGUI={self.config.ABAQUS_SCRIPT_NAME}",
                    "--", 
                    "--depth", str(depth), # depth 파라미터 전달
                    "--fidelity", str(fidelity_bo),
                    "--job_name", job_name,
                    "--work_dir", self.config.ABAQUS_WORKING_DIR
                ]
                
                subprocess.run(cmd, cwd=self.config.ABAQUS_SCRIPT_FOLDER, capture_output=True, text=True, check=True, timeout=3600)

                with open(result_file_path, 'r') as f:
                    output_value = float(f.readline().strip())
                objectives[i, 0] = output_value

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
                print(f"    ! ABAQUS call failed for {job_name}: {e}")
                objectives[i, 0] = float('nan')
            
            costs[i, 0] = time.time() - start_time
            status_msg = "Failed" if torch.isnan(objectives[i,0]) else f"Result: {objectives[i,0].item():.4e}"
            print(f"    < Evaluation {status_msg}. Time: {costs[i,0]:.2f}s")

        nan_mask = torch.isnan(objectives)
        if nan_mask.any():
            objectives[nan_mask] = self.nan_penalty
        
        is_hf = (torch.abs(X_full_norm[:, self.fidelity_dim_idx] - self.config.TARGET_FIDELITY_VALUE) < 1e-6)
        if self.negate:
            objectives[is_hf] = -objectives[is_hf]

        return objectives, costs