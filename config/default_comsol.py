# config/default_comsol.py

import os
import torch

# --- 실험 기본 정보 ---
EXPERIMENT_NAME = "wrinkle_comsol_mfbo"
SIMULATION_TOOL = "comsol"
# 동적으로 로드할 Problem 클래스의 전체 경로
PROBLEM_CLASS_PATH = "comsol_interface.problem_definition.ComsolProblem"

# --- 경로 설정 (사용자 환경에 맞게 수정 필요) ---
# 이 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MATLAB 스크립트가 있는 폴더 경로
MATLAB_SCRIPT_PATH = r"C:\Users\user\Desktop\이승원 연참" 
# COMSOL 설치 경로들
COMSOL_MLI_PATH = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\mli"
COMSOL_JRE_PATH = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\java\win64\jre"
COMSOL_SERVER_EXE = r"C:\Program Files\COMSOL\COMSOL63\Multiphysics\bin\win64\comsol.exe"

# --- 설계 변수 경계 ---
# 이전 코드와 동일하게 설정 (주름 문제에 맞게 조정 필요 시 변경)
# 예시: 종횡비 (alpha), 폭-두께 비율 (Wo/to)
ALPHA_BOUNDS = (1.0, 5.0)
TH_W_RATIO_BOUNDS = (100.0, 10000.0) # Wo/to 이므로 1/ratio 값 아님

# --- 물리적 파라미터 ---
# MATLAB 함수에 전달할 추가 인자
TARGET_STRAIN_PERCENTAGE = 15.0
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

# --- BoTorch가 사용할 정규화된 경계 (자동 생성) ---
# 설계 변수 2개 + 충실도 1개 = 3차원
NORMALIZED_BOUNDS = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)