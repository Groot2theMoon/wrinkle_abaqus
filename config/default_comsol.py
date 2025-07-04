import os
import torch

# --- 실험 기본 정보 ---
EXPERIMENT_NAME = "wrinkle_comsol_mfbo"
SIMULATION_TOOL = "comsol"
PROBLEM_CLASS_PATH = "comsol_interface.problem_definition.ComsolProblem"

# --- 경로 설정  ---
COMSOL_SERVER_PORT = 2036
COMSOL_SERVER_STARTUP_WAIT_TIME = 40
# 이 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATLAB_SCRIPT_PATH = r"C:\Users\user\Desktop\이승원 연참" 
COMSOL_MLI_PATH = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli"
COMSOL_JRE_PATH = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\java\win64\jre"
COMSOL_SERVER_EXE = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsol.exe"

# --- 설계 변수 경계 ---
ALPHA_BOUNDS = (1.0, 5.0)
TH_W_RATIO_BOUNDS = (100.0, 10000.0) # Wo/to 이므로 1/ratio 값 아님

# --- 물리적 파라미터 ---
TARGET_STRAIN_PERCENTAGE = 20.0
MATLAB_FUNCTION_NAME = "run4"

# --- BO 파라미터 ---
N_LF_INIT = 10
N_HF_INIT = 5
NUM_BO_ITERATIONS = 25
BATCH_SIZE = 1

# --- 최적화 및 모델 설정 ---
NUM_RESTARTS = 10
RAW_SAMPLES = 128
TARGET_FIDELITY_VALUE = 1.0

# --- 비용 모델 폴백 값 ---
FALLBACK_COST_LF = 60.0    # 1분
FALLBACK_COST_HF = 600.0   # 10분

# --- BoTorch가 사용할 정규화된 경계  ---
NORMALIZED_BOUNDS = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

# --- 분석 파일 이름 ---
GROUND_TRUTH_FILE_NAME = "ground_truth_comsol.csv" # ABAQUS와 다른 파일 이름

# --- 그리드 해상도 ---
GRID_RESOLUTION = 10 