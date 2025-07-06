import os
import torch
import matlab.engine
import numpy as np
import time
import subprocess
import signal

class ComsolProblem:
    """
    COMSOL/MATLAB과의 인터페이스.
    리소스 관리는 start_resources와 cleanup 클래스 메소드를 통해 명시적으로 이루어집니다.
    """
    _eng = None
    _comsol_server_process = None
    _config_ref = None # 설정을 저장할 클래스 변수

    def __init__(self, negate=True, config=None):
        if config is None:
            raise ValueError("Configuration object must be provided.")
        
        # 인스턴스가 생성될 때, 클래스 변수에 config가 저장되었는지 확인
        if ComsolProblem._config_ref is None:
            ComsolProblem._config_ref = config

        self.negate = negate
        self.config = config
        
        self._bounds = [config.ALPHA_BOUNDS, config.TH_W_RATIO_BOUNDS]
        self.num_design_vars = len(self._bounds)
        self.dim = self.num_design_vars + 1
        self.fidelity_dim_idx = self.num_design_vars
        self.nan_penalty = -1e10 if self.negate else 1e10
        self.target_fidelity_bo = config.TARGET_FIDELITY_VALUE

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
            objectives[is_hf] = -objectives[is_hf]

        return objectives, costs

    @classmethod
    def start_resources(cls, config):
        """서버와 엔진을 시작합니다. 이미 실행 중이면 아무것도 하지 않습니다."""
        print("\n" + "="*50)
        print("      STARTING COMSOL & MATLAB RESOURCES")
        print("="*50 + "\n")
        cls._config_ref = config # 항상 최신 config로 업데이트
        
        # COMSOL 서버 시작
        if cls._comsol_server_process is None or cls._comsol_server_process.poll() is not None:
            print(">>> Starting COMSOL server...")
            try:
                cmd = [config.COMSOL_SERVER_EXE, "mphserver", f"-port{config.COMSOL_SERVER_PORT}"]
                cls._comsol_server_process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
                    text=True, errors='ignore'
                )
                print(f"    COMSOL server process started (PID: {cls._comsol_server_process.pid}). Waiting for startup...")
                time.sleep(config.COMSOL_SERVER_STARTUP_WAIT_TIME)
                print("    COMSOL server ready.")
            except Exception as e:
                print(f"FATAL: COMSOL server startup failed: {e}")
                raise

        # MATLAB 엔진 시작 및 연결
        if cls._eng is None:
            print(">>> Starting MATLAB engine...")
            try:
                if os.path.isdir(config.COMSOL_JRE_PATH):
                    os.environ['MATLAB_JAVA'] = config.COMSOL_JRE_PATH
                cls._eng = matlab.engine.start_matlab()
                cls._eng.addpath(config.MATLAB_SCRIPT_PATH, nargout=0)
                connect_cmd = f"import com.comsol.model.util.*; ModelUtil.connect('localhost', {config.COMSOL_SERVER_PORT});"
                cls._eng.eval(connect_cmd, nargout=0)
                print("    MATLAB engine ready and connected to COMSOL server.")
            except Exception as e:
                print(f"FATAL: MATLAB/COMSOL connection failed: {e}")
                cls.cleanup() # 실패 시 정리
                raise
    
    @classmethod
    def cleanup(cls):
        """모든 COMSOL/MATLAB 리소스를 정리합니다."""
        print("\n" + "="*50)
        print("      CLEANING UP COMSOL & MATLAB RESOURCES")
        print("="*50 + "\n")
        if cls._eng:
            try:
                cls._eng.eval("ModelUtil.disconnect;", nargout=0)
            except Exception: pass
            finally:
                cls._eng.quit()
                cls._eng = None
                print(">>> MATLAB engine quit.")

        if cls._comsol_server_process and cls._comsol_server_process.poll() is None:
            try:
                if os.name == 'nt':
                    os.kill(cls._comsol_server_process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    cls._comsol_server_process.terminate()
                cls._comsol_server_process.wait(timeout=15)
            except Exception:
                cls._comsol_server_process.kill()
            finally:
                cls._comsol_server_process = None
                print(">>> COMSOL server stopped.")
        
        cls._config_ref = None # 참조 정리