# abaqus_run_parabolic.py (Final Corrected Version)

import os
import sys
import argparse
import traceback
import time

def create_base_model(mdb, depth, fidelity):
    """지정된 depth와 fidelity로 Abaqus 파트, 재료, 섹션, 메쉬 생성"""
    from abaqusConstants import (THREE_D, DEFORMABLE_BODY, ON, FINER, UNIFORM, 
                                  MIDDLE_SURFACE, FROM_SECTION, NEO_HOOKE)
    
    SHEET_LENGTH, SHEET_WIDTH, SHEET_THICKNESS = 0.25, 0.1, 0.0001
    num_elements_curve = 100 if fidelity == 1.0 else 40 

    model_name = 'Model'
    if model_name in mdb.models:
        del mdb.models[model_name]
    
    model = mdb.Model(name=model_name)
    s = model.ConstrainedSketch(name='__profile__', sheetSize=1.0)
    
    p1 = (0.0, -SHEET_WIDTH / 2.0); p2 = (SHEET_LENGTH, -SHEET_WIDTH / 2.0)
    p3 = (SHEET_LENGTH, SHEET_WIDTH / 2.0); p4 = (0.0, SHEET_WIDTH / 2.0)
    p_mid_bottom = (SHEET_LENGTH / 2.0, -SHEET_WIDTH / 2.0 - depth)
    p_mid_top = (SHEET_LENGTH / 2.0, SHEET_WIDTH / 2.0 + depth)
    s.Spline(points=(p2, p_mid_bottom, p1)); s.Line(point1=p1, point2=p4)
    s.Spline(points=(p4, p_mid_top, p3)); s.Line(point1=p3, point2=p2)
    
    p = model.Part(name='SheetPart', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseShell(sketch=s)
    
    material = model.Material(name='HyperelasticMat')
    material.Hyperelastic(type=NEO_HOOKE, table=((3e6, 0.0),))
    model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=SHEET_THICKNESS)
    
    faces = p.faces.getSequenceFromMask(mask=('[#1 ]',), )
    region = p.Set(faces=faces, name='AllFaces')
    p.SectionAssignment(region=region, sectionName='SheetSection', offsetType=MIDDLE_SURFACE, thicknessAssignment=FROM_SECTION)
    
    a = model.rootAssembly
    a.Instance(name='SheetInst', part=p, dependent=ON)
    
    e = p.edges
    curved_edges = e.getByBoundingBox(xMin=0.001, xMax=SHEET_LENGTH-0.001)
    p.seedEdgeByNumber(edges=curved_edges, number=num_elements_curve, constraint=FINER)
    p.generateMesh()
    
    return model

def run_linear_buckling(model, job_name, log_func):
    """선형 좌굴 해석(LF)을 수행하고 임계 하중 계수를 반환합니다."""
    from abaqus import mdb, session
    from abaqusConstants import (INITIAL, LANCZOS, UNIFORM, ON, OFF, PERCENTAGE,
                                  PERTURBATION_AND_BUCKLING)
    
    log_func(f"Starting linear buckling for job: {job_name}")
    try:
        # --- 최종 수정: Preload 스텝에서 불필요한 비선형성 제거 ---
        # 기준 응력이 0인 상태를 만들므로, nlgeom=OFF (선형 해석)로 충분하며 더 안정적임.
        model.StaticStep(name='Preload', previous='Initial', nlgeom=OFF)
        # -----------------------------------------------------------
        
        model.BuckleStep(name='Buckle', previous='Preload', numEigen=1, eigensolver=LANCZOS)
        
        a = model.rootAssembly
        inst = a.instances['SheetInst']
        
        edges1 = inst.edges.getByBoundingBox(xMin=-0.001, xMax=0.001)
        region_fix = a.Set(edges=edges1, name='Fix_Edges')
        model.EncastreBC(name='FixLeft', createStepName='Initial', region=region_fix)

        edges2 = inst.edges.getByBoundingBox(xMin=0.25-0.001, xMax=0.25+0.001)
        region_disp = a.Set(edges=edges2, name='Disp_Edges')

        # Preload 스텝의 BC. 하중이 없으므로 값은 중요하지 않지만, 강체 운동 방지를 위해 모두 0으로 구속.
        model.DisplacementBC(name='DisplacementLoad', createStepName='Preload',
                             region=region_disp, u1=0.0, u2=0.0, u3=0.0, 
                             ur1=0.0, ur2=0.0, ur3=0.0, distributionType=UNIFORM)

        # Buckle 스텝에서 좌굴을 유발할 단위 하중(Perturbation Load)을 적용
        model.boundaryConditions['DisplacementLoad'].setValuesInStep(
            stepName='Buckle', u1=-0.0025, buckleCase=PERTURBATION_AND_BUCKLING)
        
        job = mdb.Job(name=job_name, model=model.name, numCpus=1, memory=90, memoryUnits=PERCENTAGE)
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()
        
        # --- (이하 결과 추출 코드는 동일) ---
        odb_path = f'{job_name}.odb'
        if not os.path.exists(odb_path):
             log_func(f"!!! CRITICAL ERROR: ODB file not found: {odb_path}.")
             return None

        odb = session.openOdb(name=odb_path)
        critical_load_factor = None
        if 'Buckle' in odb.steps and len(odb.steps['Buckle'].frames) > 1:
            description = odb.steps['Buckle'].frames[1].description
            if 'Eigenvalue' in description:
                critical_load_factor = float(description.split('=')[1].strip())
                log_func(f"Linear buckling successful. Critical load factor: {critical_load_factor}")
            else:
                log_func(f"!!! WARNING: 'Eigenvalue' not found in frame description: {description}")
        else:
            log_func(f"!!! WARNING: 'Buckle' step not found in ODB. Preload step likely failed to converge.")

        odb.close()
        return critical_load_factor
    except Exception as e:
        log_func(f"!!! ERROR during linear buckling for {job_name}: {e}\n{traceback.format_exc()}")
        return None

def run_post_buckling(model, job_name, buckle_job_name, log_func):
    """후좌굴 해석(HF)을 수행하고 최종 주름 진폭을 반환합니다."""
    from abaqus import mdb, session
    from abaqusConstants import ON, INITIAL, UNSET, OFF, PERCENTAGE
    
    log_func(f"Starting post-buckling for job: {job_name}, using imperfection from {buckle_job_name}.odb")
    if not os.path.exists(f'{buckle_job_name}.odb'):
        log_func(f"!!! CRITICAL ERROR: Prerequisite ODB file not found: {buckle_job_name}.odb.")
        return None
    try:
        model.StaticRiksStep(name='RiksStep', previous='Initial', nlgeom=ON,
                             maxNumInc=200, initialArcInc=0.01, maxArcInc=0.1)
        
        a = model.rootAssembly
        inst = a.instances['SheetInst']

        edges1 = inst.edges.getByBoundingBox(xMin=-0.001, xMax=0.001)
        region_fix = a.Set(edges=edges1, name='Fix_Edges_Riks')
        model.EncastreBC(name='FixLeft_Riks', createStepName='Initial', region=region_fix)

        edges2 = inst.edges.getByBoundingBox(xMin=0.25-0.001, xMax=0.25+0.001)
        region_disp = a.Set(edges=edges2, name='Disp_Edges_Riks')
        
        # --- FIX: Riks 해석에서도 강체 운동을 막기 위해 U2, U3, 회전 자유도 구속 ---
        model.DisplacementBC(name='MoveRight_Riks', createStepName='RiksStep',
                             region=region_disp, u1=-0.005, u2=0.0, u3=0.0,
                             ur1=0.0, ur2=0.0, ur3=0.0, amplitude=UNSET)
        # --------------------------------------------------------------------------

        model.Imperfection(file=buckle_job_name, step=1, data=((1, 0.01 * 0.0001),))
        
        job = mdb.Job(name=job_name, model=model.name, numCpus=1, memory=90, memoryUnits=PERCENTAGE)
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()
        
        odb = session.openOdb(name=f'{job_name}.odb')
        wrinkle_amplitude = 0.0
        if 'RiksStep' in odb.steps and len(odb.steps['RiksStep'].frames) > 0:
            last_frame = odb.steps['RiksStep'].frames[-1]
            u_field = last_frame.fieldOutputs['U']
            u3_values = [v.data[2] for v in u_field.values]
            if u3_values:
                wrinkle_amplitude = max(u3_values) - min(u3_values)
            log_func(f"Post-buckling successful. Wrinkle amplitude: {wrinkle_amplitude}")
        else:
            log_func(f"!!! WARNING: No valid frames found in RiksStep for job {job_name}.")

        odb.close()
        return wrinkle_amplitude
    except Exception as e:
        log_func(f"!!! ERROR during post-buckling for {job_name}: {e}\n{traceback.format_exc()}")
        return None


def main():
    """스크립트의 메인 실행 함수. 인자 파싱 및 해석 흐름을 제어합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, required=True)
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--fidelity", type=float, required=True)
    args, _ = parser.parse_known_args(args=sys.argv)
    
    sanitized_job_name = args.job_name.replace('.', 'p')
    
    work_dir = os.getcwd()
    log_file_path = os.path.join(work_dir, f"log_{sanitized_job_name}.txt")
    
    def write_log(message):
        with open(log_file_path, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    try:
        write_log("Attempting to import Abaqus modules...")
        from abaqus import mdb, session
        write_log("Abaqus modules imported successfully.")

        result_value = None
        if args.fidelity == 0.0:
            write_log("Running LF Analysis (Displacement Control)")
            model = create_base_model(mdb, args.depth, args.fidelity)
            result_value = run_linear_buckling(model, sanitized_job_name, write_log)
        else: # fidelity == 1.0
            write_log("Running HF Analysis")
            buckle_job_name = f"{sanitized_job_name}_buckle"
            buckle_model = create_base_model(mdb, args.depth, 0.0)
            buckling_load_factor = run_linear_buckling(buckle_model, buckle_job_name, write_log)
            
            if buckling_load_factor is not None and buckling_load_factor > 0:
                post_model = create_base_model(mdb, args.depth, 1.0)
                result_value = run_post_buckling(post_model, sanitized_job_name, buckle_job_name, write_log)
            else:
                log_func_message = (f"Skipping post-buckling because linear buckling "
                                    f"did not yield a positive load factor (got: {buckling_load_factor}).")
                write_log(log_func_message)
        
        if result_value is not None:
            result_file_path = os.path.join(work_dir, f"result_{sanitized_job_name}.txt")
            with open(result_file_path, 'w') as f: f.write(str(result_value))
            write_log(f"Success! Result {result_value} written to {result_file_path}")
        else:
            write_log("Analysis failed or returned no valid result. No result file created.")
            
    except Exception as e:
        write_log(f"!!! CATASTROPHIC ERROR in main function: {e}\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()