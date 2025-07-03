# config/default.py
import os
import torch

# --- 실험 기본 정보 ---
EXPERIMENT_NAME = "wrinkle_mfbo_default"

# --- 경로 설정 (사용자 환경에 맞게 수정 필요) ---
ABAQUS_EXE_PATH = r"C:\SIMULIA\Commands\abaqus.exe"
ABAQUS_SCRIPT_NAME = "run_abaqus_analysis.py"
# __file__을 이용해 현재 config 파일의 위치를 기준으로 경로를 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABAQUS_SCRIPT_FOLDER = PROJECT_ROOT
ABAQUS_WORKING_DIR = os.path.join(PROJECT_ROOT, "abaqus_workspace")

# --- 분석 파일 ---
GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, "ground_truth.csv")

# --- 설계 변수 경계 ---
# (종횡비, 두께-폭 비율)
ALPHA_BOUNDS = (1.0, 5.0)
TH_W_RATIO_BOUNDS = (100.0, 10000.0)

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

# --- BoTorch가 사용할 정규화된 경계 (자동 생성) ---
# 설계 변수 2개 + 충실도 1개 = 3차원
NORMALIZED_BOUNDS = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)

GRID_RESOLUTION = 10