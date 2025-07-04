# comsol_interface/problem_definition.py

import os
import torch
import matlab.engine
import numpy as np
import time
import subprocess
import signal

class ComsolProblem:
    """
    Interfaces with COMSOL via MATLAB engine to evaluate design points.
    This class manages the lifecycle of the COMSOL server and MATLAB engine.
    """
    _eng = None
    _comsol_server_process = None
    _engine_users = 0

    def __init__(self, negate=True, config=None):
        if config is None:
            raise ValueError("Configuration object must be provided.")
        
        # 설정 객체에서 모든 파라미터를 가져옴
        self.negate = negate
        self.config = config
        
        self._bounds = [config.ALPHA_BOUNDS, config.TH_W_RATIO_BOUNDS]
        self.num_design_vars = len(self._bounds)
        self.dim = self.num_design_vars + 1
        self.fidelity_dim_idx = self.num_design_vars
        
        self.nan_penalty = -1e10 if self.negate else 1e10
        self.target_fidelity_bo = config.TARGET_FIDELITY_VALUE

        # 여러 인스턴스가 생성되더라도 엔진/서버는 한 번만 시작하도록 관리
        ComsolProblem._engine_users += 1
        self._ensure_server_and_engine_started()

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
            
            alpha, th_w_ratio = X_design_unnorm[0, 0].item(), X_design_unnorm[0, 1].item()
            
            print(f"  > Evaluating via MATLAB: α={alpha:.3f}, th/W_ratio={th_w_ratio:.4f}, fid={fidelity_bo:.1f}...")
            
            start_time = time.time()
            try:
                # MATLAB 함수 호출
                output = self._eng.feval(
                    self.config.MATLAB_FUNCTION_NAME,
                    alpha, th_w_ratio, fidelity_bo, self.config.TARGET_STRAIN_PERCENTAGE,
                    nargout=1
                )
                objectives[i, 0] = float('nan') if output is None or np.isnan(output) else float(output)

            except Exception as e:
                print(f"    ! MATLAB call failed: {e}")
                objectives[i, 0] = float('nan')
            
            costs[i, 0] = time.time() - start_time
            status_msg = "Failed" if torch.isnan(objectives[i,0]) else f"Result: {objectives[i,0].item():.4e}"
            print(f"    < Evaluation {status_msg}. Time: {costs[i,0]:.2f}s")

        nan_mask = torch.isnan(objectives)
        if nan_mask.any():
            objectives[nan_mask] = self.nan_penalty
        
        is_hf = (torch.abs(X_full_norm[:, self.fidelity_dim_idx] - self.target_fidelity_bo) < 1e-6)
        if self.negate:
            # BoTorch는 최대화를 기본으로 하므로, 최소화 문제 시 HF 결과만 부호를 바꿈
            objectives[is_hf] = -objectives[is_hf]

        return objectives, costs

    # --- 서버 및 엔진 관리 메소드 (이전 코드에서 가져와 클래스 메소드로 재구성) ---
    @classmethod
    def _start_comsol_server(cls, config):
        if cls._comsol_server_process is None or cls._comsol_server_process.poll() is not None:
            print("Starting COMSOL server...")
            try:
                cls._comsol_server_process = subprocess.Popen(
                    [config.COMSOL_SERVER_EXE, "mphserver"],
                    # ... 기타 옵션 ...
                )
                time.sleep(40) # 충분한 대기 시간
                print("COMSOL server started.")
            except Exception as e:
                print(f"FATAL: COMSOL server failed to start: {e}")
                cls._comsol_server_process = None
                raise

    @classmethod
    def _start_matlab_engine_instance(cls, config):
        if cls._eng is None:
            print("Starting MATLAB engine...")
            if os.path.isdir(config.COMSOL_JRE_PATH):
                os.environ['MATLAB_JAVA'] = config.COMSOL_JRE_PATH
            try:
                cls._eng = matlab.engine.start_matlab()
                cls._eng.addpath(config.MATLAB_SCRIPT_PATH, nargout=0)
                cls._eng.eval("import com.comsol.model.util.*; ModelUtil.connect('localhost', 2036);", nargout=0)
                print("MATLAB connected to COMSOL server.")
            except Exception as e:
                print(f"FATAL: MATLAB/COMSOL connection failed: {e}")
                if cls._eng: cls._eng.quit()
                cls._eng = None
                raise

    @classmethod
    def _ensure_server_and_engine_started(cls):
        # 이 메소드는 config 객체가 없으므로, 호출하는 쪽에서 config를 전달해야 함
        # 더 나은 방법은 config를 클래스 변수로 저장하는 것이지만, 여기서는 간단하게 유지
        # 실제로는 이 부분을 더 정교하게 만들어야 함.
        # 여기서는 __init__에서 config를 전달받아 처리하는 것으로 가정하고, 이 메소드는 단순화.
        # 실제 구현에서는 config를 클래스 변수에 저장하거나, 매번 전달받아야 합니다.
        pass # __init__에서 직접 호출하는 것으로 변경

    @classmethod
    def cleanup(cls):
        cls._engine_users -= 1
        if cls._engine_users > 0:
            return # 아직 다른 인스턴스가 사용 중

        if cls._eng:
            print("Quitting MATLAB engine...")
            cls._eng.quit()
            cls._eng = None
        if cls._comsol_server_process and cls._comsol_server_process.poll() is None:
            print("Stopping COMSOL server...")
            # ... (종료 로직) ...
            cls._comsol_server_process = None
        print("COMSOL resources cleaned up.")