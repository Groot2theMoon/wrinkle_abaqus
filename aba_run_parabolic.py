from abaqus import *
from abaqusConstants import *
from caeModules import *
import os
import sys
import argparse
import traceback

SHEET_LENGTH = 0.25
SHEET_WIDTH = 0.1
SHEET_THICKNESS = 0.0001
MESH_SIZE_LF = 5e-3  # 저충실도 메쉬 크기
MESH_SIZE_HF = 1e-3  # 고충실도 메쉬 크기

def create_base_model(model_name, depth, fidelity):

    model = mdb.Model(name=model_name)
    
    # --- 파라메트릭 스케치 및 파트 생성 ---
    sketch = model.ConstrainedSketch(name='__profile__', sheetSize=1.0)
    p1 = (0.0, -SHEET_WIDTH / 2.0)
    p2 = (SHEET_LENGTH, -SHEET_WIDTH / 2.0)
    p3 = (SHEET_LENGTH, SHEET_WIDTH / 2.0)
    p4 = (0.0, SHEET_WIDTH / 2.0)
    p_mid_bottom = (SHEET_LENGTH / 2.0, -SHEET_WIDTH / 2.0 - depth)
    p_mid_top = (SHEET_LENGTH / 2.0, SHEET_WIDTH / 2.0 + depth)

    sketch.Spline(points=(p2, p_mid_bottom, p1))
    sketch.Line(point1=p1, point2=p4)
    sketch.Spline(points=(p4, p_mid_top, p3))
    sketch.Line(point1=p3, point2=p2)
    
    part = model.Part(name='SheetPart', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)

    # --- 재료, 섹션, 어셈블리 ---
    material = model.Material(name='HyperelasticMat')
    material.Hyperelastic(type=NEO_HOOKE, table=((3e6, 0.0),))
    model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=SHEET_THICKNESS)
    region = part.faces
    part.SectionAssignment(region=region, sectionName='SheetSection')
    
    assembly = model.rootAssembly
    instance = assembly.Instance(name='SheetInst', part=part, dependent=ON)

    mesh_size = MESH_SIZE_LF if fidelity == 0.0 else MESH_SIZE_HF
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    
    return model

def run_linear_buckling(model, job_name):
    """
    선형 좌굴 해석을 수행하고 최저 임계 하중을 반환.
    """
    model.StaticPerturbationStep(name='Load', previous='Initial')
    model.BuckleStep(name='Buckle', previous='Load', numEigen=1, eigensolver=LANCZOS)

    assembly = model.rootAssembly
    instance = assembly.instances['SheetInst']
    
    left_edges = instance.edges.getByBoundingBox(xMin=-0.001, xMax=0.001, yMin=-SHEET_WIDTH, yMax=SHEET_WIDTH)
    assembly.EncastreBC(name='FixLeft', createStepName='Initial', region=left_edges)
    
    right_edges = instance.edges.getByBoundingBox(xMin=SHEET_LENGTH-0.001, xMax=SHEET_LENGTH+0.001, yMin=-SHEET_WIDTH, yMax=SHEET_WIDTH)
    assembly.ShellEdgeLoad(name='UnitLoad', createStepName='Load', region=right_edges, 
                           magnitude=1.0, distributionType=UNIFORM)
    
    job = mdb.Job(name=job_name, model=model.name, description='Linear Buckling Analysis')
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    try:
        odb = session.openOdb(name=f'{job_name}.odb')
        eigenvalue_str = odb.steps['Buckle'].frames[1].description
        critical_load = float(eigenvalue_str.split('=')[1].strip())
        odb.close()
        return critical_load
    except Exception as e:
        print(f"Error extracting eigenvalue for {job_name}: {e}")
        traceback.print_exc()
        return None

def run_post_buckling(model, job_name, buckle_job_name):
    """
    선형 좌굴 해석 결과를 초기 결함으로 사용하여 후좌굴 해석을 수행.
    """
    model.StaticRiksStep(name='RiksStep', previous='Initial', maxNumInc=200, nlgeom=ON,
                         initialArcInc=0.01, maxArcInc=0.1) # Riks 파라미터 추가

    assembly = model.rootAssembly
    instance = assembly.instances['SheetInst']
    
    left_edges = instance.edges.getByBoundingBox(xMin=-0.001, xMax=0.001, yMin=-SHEET_WIDTH, yMax=SHEET_WIDTH)
    assembly.EncastreBC(name='FixLeft', createStepName='Initial', region=left_edges)
    
    right_edges = instance.edges.getByBoundingBox(xMin=SHEET_LENGTH-0.001, xMax=SHEET_LENGTH+0.001, yMin=-SHEET_WIDTH, yMax=SHEET_WIDTH)
    assembly.DisplacementBC(name='MoveRight', createStepName='RiksStep', region=right_edges, 
                            u1=-0.0025, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)

    model.Imperfection(
        file=buckle_job_name, 
        step=1,             
        data=((1, 0.01 * SHEET_THICKNESS),) 
    )

    job = mdb.Job(name=job_name, model=model.name, description='Post-Buckling Analysis')
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()

    try:
        odb = session.openOdb(name=f'{job_name}.odb')
        riks_step = odb.steps['RiksStep']
        last_completed_frame = None
        for frame in reversed(riks_step.frames):
            if frame.description.startswith("Increment") and "Step Time" in frame.description:
                last_completed_frame = frame
                break
        
        if last_completed_frame:
            u_field = last_completed_frame.fieldOutputs['U']
            u3_values = [v.data[2] for v in u_field.values]
            wrinkle_amplitude = max(u3_values) - min(u3_values) if u3_values else 0.0
        else:
            wrinkle_amplitude = 0.0
        
        odb.close()
        return wrinkle_amplitude
    except Exception as e:
        print(f"Error extracting results for {job_name}: {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':

    if '--' in sys.argv:
        args_list = sys.argv[sys.argv.index('--') + 1:]
    else:
        args_list = []

    parser = argparse.ArgumentParser(description="Multi-fidelity wrinkling analysis script for Abaqus.")
    parser.add_argument("--depth", type=float, required=True, help="Initial curvature depth of the sheet.")
    parser.add_argument("--fidelity", type=float, required=True, choices=[0.0, 1.0], help="Fidelity level (0.0 for LF, 1.0 for HF).")
    parser.add_argument("--job_name", type=str, required=True, help="Base name for the analysis job.")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory for files.")
    args = parser.parse_args(args_list)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    os.chdir(args.work_dir)
    
    Mdb()

    result_value = None

    if args.fidelity == 0.0:
        print("--- Running Linear Buckling Analysis (LF) ---")

        lf_model = create_base_model(model_name='LF_Model', depth=args.depth, fidelity=args.fidelity)
        lf_job_name = f"{args.job_name}_lf"
        result_value = run_linear_buckling(lf_model, lf_job_name)

    elif args.fidelity == 1.0:
        print("--- Running Post-Buckling Analysis (HF) ---")
        
        buckle_model_name = 'HF_Buckle_Model'
        buckle_job_name = f"{args.job_name}_hf_buckle"
        buckle_model = create_base_model(model_name=buckle_model_name, depth=args.depth, fidelity=0.0) 
        run_linear_buckling(buckle_model, buckle_job_name)
        
        post_model_name = 'HF_Post_Model'
        post_buckle_model = create_base_model(model_name=post_model_name, depth=args.depth, fidelity=args.fidelity)
        
        hf_job_name = f"{args.job_name}_hf_post"
        result_value = run_post_buckling(
            model=post_buckle_model,
            job_name=hf_job_name,
            buckle_job_name=buckle_job_name, 
        )

    if result_value is not None:
        result_file_path = os.path.join(os.getcwd(), f"result_{args.job_name}.txt")
        with open(result_file_path, 'w') as f:
            f.write(str(result_value))
        print(f"Successfully wrote result: {result_value} to {result_file_path}")
    else:
        print(f"Analysis failed or result extraction failed for job {args.job_name}.")

        result_file_path = os.path.join(os.getcwd(), f"result_{args.job_name}_FAILED.txt")
        with open(result_file_path, 'w') as f:
            f.write("Analysis Failed")