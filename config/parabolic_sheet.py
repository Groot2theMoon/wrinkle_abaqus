import torch
import os

# ----------------- 실험 기본 설정 -----------------
EXPERIMENT_NAME = "Parabolic_Wrinkle_1D_v1"
PROBLEM_CLASS_PATH = "abaqus_interface.parabolic_sheet.AbaqusParabolicWrinkle"

# ----------------- 아바쿠스 관련 설정 -----------------
ABAQUS_EXE_PATH = r"C:\SIMULIA\Commands\abaqus.exe"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABAQUS_SCRIPT_FOLDER = PROJECT_ROOT
ABAQUS_WORKING_DIR = os.path.join(PROJECT_ROOT, "abaqus_workspace")
ABAQUS_SCRIPT_NAME = "aba_run_parabolic.py" 

# ----------------- BO  -----------------
N_LF_INIT = 10
N_HF_INIT = 5
NUM_BO_ITERATIONS = 15  

BATCH_SIZE = 1          
NUM_RESTARTS = 10      
RAW_SAMPLES = 256       

# ----------------- 설계 변수 및 경계 -----------------
DEPTH_BOUNDS = [0.1, 5.0]

NORMALIZED_BOUNDS = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

# ----------------- 충실도 -----------------
TARGET_FIDELITY_VALUE = 1.0 

FALLBACK_COST_LF = 60.0    # LF 해석은 1분 정도 걸린다고 가정
FALLBACK_COST_HF = 600.0   # HF 해석은 10분 정도 걸린다고 가정

# ----------------- 분석용 Ground Truth 파일 -----------------
GROUND_TRUTH_FILE = "path/to/your/ground_truth_1d.csv"

GRID_RESOLUTION = 10  # 그리드 해상도