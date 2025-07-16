import os
import sys
import argparse
import traceback
import time
import subprocess
import glob
import numpy as np

try:
    from abaqus import mdb
    from abaqusConstants import (ON, OFF, THREE_D, DEFORMABLE_BODY, NEO_HOOKE, 
                                LANCZOS, MIDDLE_SURFACE, ISOTROPIC, 
                                ANALYSIS, VOLUMETRIC_DATA, DEFAULT)
    from odbAccess import openOdb
except ImportError:
    print("WARNING: Abaqus modules not found. This script must be run within the Abaqus Python environment.")

ANALYSIS_PARAMETERS = {
    "SHEET_LENGTH": 0.25,
    "SHEET_WIDTH": 0.1,
    "SHEET_THICKNESS": 0.0001,
    "MATERIAL_PROPS": ((3e6, 0.0),),
    "APPLIED_DISPLACEMENT_BUCKLE": 0.0025,
    "APPLIED_DISPLACEMENT_RIKS": 0.0025,
    "MESH_SIZE_PRE_BUCKLE": 0.002,
    "MESH_SIZE_POST_BUCKLE": 0.001,
    "INITIAL_INCREMENT_SIZE": 1.0,
    "RIKS_MAX_INCREMENTS": 200,
    "RIKS_INITIAL_ARC_INC": 1,
    "ABAQUS_TIMEOUT_SECONDS": 1000,
    "RIKS_MAX_ARC_INC":1.0
    #"PRE_DISP" : 0.0001
}

def cleanup_abaqus_files(base_job_name):
    print(f"Cleaning up files for job base: {base_job_name}...")
    files_to_delete = glob.glob(f"{base_job_name}*")
    
    for f_path in files_to_delete:
        if not f_path.endswith(('.py', '.csv', '.txt', '.inp', '.odb', '.msg')):
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"  - Error deleting file {f_path}: {e}")

def write_result_to_txt(job_name, result_value):
    file_path = f"result_{job_name}.txt"
    try:
        with open(file_path, 'w') as f:
            f.write(str(result_value))
    except IOError as e:
        print(f"Error writing to result TXT file {file_path}: {e}")

def create_base_model(model_name, depth, mesh_size):
    model = mdb.Model(name=model_name)
    sheet_length = ANALYSIS_PARAMETERS["SHEET_LENGTH"]
    sheet_width = ANALYSIS_PARAMETERS["SHEET_WIDTH"]
    sketch = model.ConstrainedSketch(name='__profile__', sheetSize=1.0)

    p1 = (0.0, -sheet_width / 2.0); p2 = (sheet_length, -sheet_width / 2.0); p3 = (sheet_length, sheet_width / 2.0); p4 = (0.0, sheet_width / 2.0)
    p_mid_bottom = (sheet_length / 2.0, -sheet_width / 2.0 - depth); p_mid_top = (sheet_length / 2.0, sheet_width / 2.0 + depth)
    sketch.Spline(points=(p2, p_mid_bottom, p1)); sketch.Line(point1=p1, point2=p4); sketch.Spline(points=(p4, p_mid_top, p3)); sketch.Line(point1=p3, point2=p2)
    
    part = model.Part(name='SheetPart', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)

    material = model.Material(name='HyperelasticMat')
    material.Hyperelastic(materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, volumetricResponse=VOLUMETRIC_DATA, table=ANALYSIS_PARAMETERS["MATERIAL_PROPS"])
    model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=ANALYSIS_PARAMETERS["SHEET_THICKNESS"])
    
    all_faces = part.faces.getSequenceFromMask(mask=('[#1 ]',), )
    part.SectionAssignment(region=part.Set(faces=all_faces, name='AllFaces'), sectionName='SheetSection', offsetType=MIDDLE_SURFACE)
    
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()

    fix_edges_region = part.edges.getByBoundingBox(xMin=-0.001, xMax=0.001)
    part.Set(edges=fix_edges_region, name='Fix_Edges')

    disp_edges_region = part.edges.getByBoundingBox(xMin=sheet_length - 0.001, xMax=sheet_length + 0.001)
    part.Set(edges=disp_edges_region, name='Disp_Edges')

    assembly = model.rootAssembly
    assembly.Instance(name='SheetInst', part=part, dependent=ON)
    return model

def run_linear_buckle_analysis(job_name, depth):

    if depth > 0.01:
        PRE_DISP = 0.000025
    else:
        PRE_DISP = 0.000125

    model = create_base_model(model_name='Buckle_Model', depth=depth, mesh_size=ANALYSIS_PARAMETERS["MESH_SIZE_PRE_BUCKLE"])
    instance = model.rootAssembly.instances['SheetInst']
    model.StaticStep(name='Preload', previous='Initial', nlgeom=ON, initialInc=ANALYSIS_PARAMETERS["INITIAL_INCREMENT_SIZE"])
    model.BuckleStep(name='Buckle', previous='Preload', numEigen=1, maxEigen=100.0, eigensolver=LANCZOS, minEigen=0.0, blockSize=DEFAULT, maxBlocks=DEFAULT)
    model.EncastreBC(name='FixLeft', createStepName='Initial', region=instance.sets['Fix_Edges'])
    bc = model.DisplacementBC(name='LoadBC', createStepName='Preload', region=instance.sets['Disp_Edges'], u1=PRE_DISP, u2=0, u3=0, ur1=0, ur2=0, ur3=0)
    bc.setValuesInStep(stepName='Buckle', u1=ANALYSIS_PARAMETERS["APPLIED_DISPLACEMENT_BUCKLE"])
    job = mdb.Job(name=job_name, model=model.name)
    job.submit(consistencyChecking=OFF); job.waitForCompletion()
    odb_path = f'{job_name}.odb'
    eigenvalue = "NaN"
    if os.path.exists(odb_path):
        try:
            odb = openOdb(path=odb_path, readOnly=True)
            if 'Buckle' in odb.steps and len(odb.steps['Buckle'].frames) > 1:
                description = odb.steps['Buckle'].frames[1].description
                eigenvalue = float(description.split('=')[1].strip())
            odb.close()
        except Exception: print(f"Warning: ODB access error for {job_name}.")
    
    
    result = PRE_DISP + ANALYSIS_PARAMETERS["APPLIED_DISPLACEMENT_BUCKLE"] * eigenvalue
    
    return result
    #return np.log(result + 1e-12) 

def run_post_buckle_analysis(job_name, depth, buckle_job_name):
    model = create_base_model(model_name='Post_Model', depth=depth, mesh_size=ANALYSIS_PARAMETERS["MESH_SIZE_POST_BUCKLE"])
    instance = model.rootAssembly.instances['SheetInst']
    model.StaticRiksStep(name='Riks', previous='Initial', nlgeom=ON, maxNumInc=ANALYSIS_PARAMETERS["RIKS_MAX_INCREMENTS"], initialArcInc=ANALYSIS_PARAMETERS["RIKS_INITIAL_ARC_INC"], maxArcInc=ANALYSIS_PARAMETERS["RIKS_MAX_ARC_INC"])
    model.EncastreBC(name='FixLeft', createStepName='Initial', region=instance.sets['Fix_Edges'])
    model.DisplacementBC(name='Riks_Disp', createStepName='Riks', region=instance.sets['Disp_Edges'], u1=ANALYSIS_PARAMETERS["APPLIED_DISPLACEMENT_RIKS"], u2=0, u3=0, ur1=0, ur2=0, ur3=0)
    temp_inp_job_name = f"{job_name}_temp_for_inp"
    mdb.Job(name=temp_inp_job_name, model=model.name, type=ANALYSIS).writeInput(consistencyChecking=OFF)
    inp_path = f"{temp_inp_job_name}.inp"
    with open(inp_path, 'r') as f_in: lines = f_in.readlines()
    with open(inp_path, 'w') as f_out:
        for line in lines:
            if line.strip().lower().startswith('*step'):
                imperfection_scale = 0.001 #ANALYSIS_PARAMETERS["SHEET_THICKNESS"] * 0.05
                imperfection_string = f"*IMPERFECTION, FILE={buckle_job_name}, STEP=2\n1, {imperfection_scale}\n"
                f_out.write(imperfection_string)
            f_out.write(line)
    command = f"abaqus job={job_name} input={inp_path} interactive"
    process = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=ANALYSIS_PARAMETERS["ABAQUS_TIMEOUT_SECONDS"])
    if process.returncode != 0: print(f"Warning: Abaqus execution failed for {job_name}. Stderr: {process.stderr}")
    odb_path = f'{job_name}.odb'
    result_value = "NaN"
    if os.path.exists(odb_path):
        try:
            odb = openOdb(path=odb_path, readOnly=True)
            if 'Riks' in odb.steps and len(odb.steps['Riks'].frames) > 0:
                last_frame = odb.steps['Riks'].frames[-1]
                if 'U' in last_frame.fieldOutputs:
                    u_field = last_frame.fieldOutputs['U']
                    u3_values = [v.data[2] for v in u_field.values if hasattr(v, 'data')]
                    if u3_values: result_value = max(u3_values) - min(u3_values)
            odb.close()
        except Exception: print(f"Warning: Post-buckle ODB access error for {job_name}.")
    return result_value

def main():
    parser = argparse.ArgumentParser(description="Run Abaqus Buckling Analysis and clean up files.")
    parser.add_argument("--job_name", required=True, help="Base name for the Abaqus job.")
    parser.add_argument("--depth", type=float, required=True, help="Curvature depth of the sheet.")
    parser.add_argument("--fidelity", type=float, required=True, help="Analysis fidelity (0.0 for LF, 1.0 for HF).")
    args, _ = parser.parse_known_args()

    result_value = "NaN"

    try:
        print(f"--- Starting Analysis for {args.job_name} ---")
        if args.fidelity == 0.0:
            result_value = run_linear_buckle_analysis(job_name=args.job_name, depth=args.depth)
        elif args.fidelity == 1.0:
            buckle_job_name = f"{args.job_name}_pre_buckle"
            buckling_load_factor = run_linear_buckle_analysis(job_name=buckle_job_name, depth=args.depth)
            if isinstance(buckling_load_factor, float) and buckling_load_factor > 0:
                result_value = run_post_buckle_analysis(job_name=args.job_name, depth=args.depth, buckle_job_name=buckle_job_name)
            else:
                print(f"Pre-buckle analysis failed. Skipping post-buckle analysis.")
    except Exception as e:
        print(f"ERROR during analysis for {args.job_name}: {e}\n{traceback.format_exc()}")
    finally:
        write_result_to_txt(args.job_name, result_value)
        cleanup_abaqus_files(args.job_name)
        print(f"--- Analysis for {args.job_name} Finished. Result: {result_value} ---")

if __name__ == '__main__':
    main()