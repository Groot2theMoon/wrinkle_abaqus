import os
import sys
import argparse
import traceback
import time
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--fidelity", type=float, required=True)
    args, _ = parser.parse_known_args(sys.argv)
    
    log_file_path = f"log_{args.job_name}.txt"
    def write_log(message):
        with open(log_file_path, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    try:
        from abaqus import mdb, session
        from abaqusConstants import ON, OFF, THREE_D, DEFORMABLE_BODY, NEO_HOOKE, LANCZOS, FINER, MIDDLE_SURFACE, ISOTROPIC, ANALYSIS, VOLUMETRIC_DATA
        from odbAccess import openOdb

        result_value = "NaN" 

        if args.fidelity == 0.0:
            # --- LF 해석 (변경 없음) ---
            model_name = 'LF_Model'
            job_name = args.job_name
            write_log(f"Running LF Buckle Analysis for {job_name}.")
            model = mdb.Model(name=model_name)
            s = model.ConstrainedSketch(name='__profile__', sheetSize=1.0)
            SHEET_LENGTH, SHEET_WIDTH, SHEET_THICKNESS = 0.25, 0.1, 0.0001
            p1 = (0.0, -SHEET_WIDTH / 2.0); p2 = (SHEET_LENGTH, -SHEET_WIDTH / 2.0); p3 = (SHEET_LENGTH, SHEET_WIDTH / 2.0); p4 = (0.0, SHEET_WIDTH / 2.0)
            p_mid_bottom = (SHEET_LENGTH / 2.0, -SHEET_WIDTH / 2.0 - args.depth); p_mid_top = (SHEET_LENGTH / 2.0, SHEET_WIDTH / 2.0 + args.depth)
            s.Spline(points=(p2, p_mid_bottom, p1)); s.Line(point1=p1, point2=p4); s.Spline(points=(p4, p_mid_top, p3)); s.Line(point1=p3, point2=p2)
            p = model.Part(name='SheetPart', dimensionality=THREE_D, type=DEFORMABLE_BODY); p.BaseShell(sketch=s)
            material = model.Material(name='HyperelasticMat')
            material.Hyperelastic(materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, volumetricResponse=VOLUMETRIC_DATA, table=((3e6, 0.0), ))
            model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=SHEET_THICKNESS)
            p.SectionAssignment(region=p.Set(faces=p.faces.getSequenceFromMask(mask=('[#1 ]',), ), name='AllFaces'), sectionName='SheetSection', offsetType=MIDDLE_SURFACE)
            p.seedPart(size=0.01, deviationFactor=0.1, minSizeFactor=0.1)
            p.generateMesh()
            p.Set(edges=p.edges.getByBoundingBox(xMin=-0.001, xMax=0.001), name='Fix_Edges')
            p.Set(edges=p.edges.getByBoundingBox(xMin=SHEET_LENGTH-0.001, xMax=SHEET_LENGTH+0.001), name='Disp_Edges')
            a = model.rootAssembly; a.Instance(name='SheetInst', part=p, dependent=ON)
            model.StaticStep(name='Preload', previous='Initial', nlgeom=ON, initialInc=0.1)
            model.BuckleStep(name='Buckle', previous='Preload', numEigen=1, eigensolver=LANCZOS)
            model.EncastreBC(name='FixLeft', createStepName='Initial', region=a.instances['SheetInst'].sets['Fix_Edges'])
            bc = model.DisplacementBC(name='LoadBC', createStepName='Preload', region=a.instances['SheetInst'].sets['Disp_Edges'], u1=-1e-9, u2=0, u3=0, ur1=0, ur2=0, ur3=0)
            bc.setValuesInStep(stepName='Buckle', u1=-0.0025)
            job = mdb.Job(name=job_name, model=model.name)
            job.submit(consistencyChecking=OFF)
            job.waitForCompletion()
            odb_path = f'{job_name}.odb'
            if os.path.exists(odb_path):
                try:
                    odb = openOdb(path=odb_path, readOnly=True)
                    if 'Buckle' in odb.steps and len(odb.steps['Buckle'].frames) > 1:
                        result_value = float(odb.steps['Buckle'].frames[1].description.split('=')[1].strip())
                    odb.close()
                except Exception as odb_e:
                    write_log(f"ODB access error for {job_name}: {odb_e}")

        elif args.fidelity == 1.0:
            buckle_job_name = f"{args.job_name}_buckle"
            write_log(f"Running Pre-Buckle Analysis for {buckle_job_name}.")
            buckle_model = mdb.Model(name='Buckle_Model')
            s_b = buckle_model.ConstrainedSketch(name='__profile__', sheetSize=1.0)
            SHEET_LENGTH_b, SHEET_WIDTH_b, SHEET_THICKNESS_b = 0.25, 0.1, 0.0001
            p1_b = (0.0, -SHEET_WIDTH_b / 2.0); p2_b = (SHEET_LENGTH_b, -SHEET_WIDTH_b / 2.0); p3_b = (SHEET_LENGTH_b, SHEET_WIDTH_b / 2.0); p4_b = (0.0, SHEET_WIDTH_b / 2.0)
            p_mid_bottom_b = (SHEET_LENGTH_b / 2.0, -SHEET_WIDTH_b / 2.0 - args.depth); p_mid_top_b = (SHEET_LENGTH_b / 2.0, SHEET_WIDTH_b / 2.0 + args.depth)
            s_b.Spline(points=(p2_b, p_mid_bottom_b, p1_b)); s_b.Line(point1=p1_b, point2=p4_b); s_b.Spline(points=(p4_b, p_mid_top_b, p3_b)); s_b.Line(point1=p3_b, point2=p2_b)
            p_b = buckle_model.Part(name='SheetPart_B', dimensionality=THREE_D, type=DEFORMABLE_BODY); p_b.BaseShell(sketch=s_b)
            material_b = buckle_model.Material(name='HyperelasticMat')
            material_b.Hyperelastic(materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, volumetricResponse=VOLUMETRIC_DATA, table=((3e6, 0.0), ))
            buckle_model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=SHEET_THICKNESS_b)
            p_b.SectionAssignment(region=p_b.Set(faces=p_b.faces.getSequenceFromMask(mask=('[#1 ]',), ), name='AllFaces'), sectionName='SheetSection', offsetType=MIDDLE_SURFACE)
            p_b.seedPart(size=0.01, deviationFactor=0.1, minSizeFactor=0.1)
            p_b.generateMesh()
            p_b.Set(edges=p_b.edges.getByBoundingBox(xMin=-0.001, xMax=0.001), name='Fix_Edges')
            p_b.Set(edges=p_b.edges.getByBoundingBox(xMin=SHEET_LENGTH_b-0.001, xMax=SHEET_LENGTH_b+0.001), name='Disp_Edges')
            a_b = buckle_model.rootAssembly; a_b.Instance(name='SheetInst_B', part=p_b, dependent=ON)
            buckle_model.StaticStep(name='Preload', previous='Initial', nlgeom=ON, initialInc=0.1)
            buckle_model.BuckleStep(name='Buckle', previous='Preload', numEigen=1, eigensolver=LANCZOS)
            buckle_model.EncastreBC(name='FixLeft', createStepName='Initial', region=a_b.instances['SheetInst_B'].sets['Fix_Edges'])
            bc_b = buckle_model.DisplacementBC(name='LoadBC', createStepName='Preload', region=a_b.instances['SheetInst_B'].sets['Disp_Edges'], u1=-1e-9, u2=0, u3=0, ur1=0, ur2=0, ur3=0)
            bc_b.setValuesInStep(stepName='Buckle', u1=-0.0025)
            buckle_job = mdb.Job(name=buckle_job_name, model=buckle_model.name)
            buckle_job.submit(consistencyChecking=OFF); buckle_job.waitForCompletion()

            buckling_load_factor = -1.0
            buckle_odb_path = f'{buckle_job_name}.odb'
            if os.path.exists(buckle_odb_path):
                try:
                    odb_b = openOdb(path=buckle_odb_path, readOnly=True)
                    if 'Buckle' in odb_b.steps and len(odb_b.steps['Buckle'].frames) > 1:
                        buckling_load_factor = float(odb_b.steps['Buckle'].frames[1].description.split('=')[1].strip())
                    odb_b.close()
                except Exception as odb_e:
                    write_log(f"Buckle ODB access error for {buckle_job_name}: {odb_e}")

            if buckling_load_factor > 0:
                job_name = args.job_name
                write_log(f"Running Post-Buckle Analysis for {job_name}.")
                
                post_model = mdb.Model(name='Post_Model')
                s_p = post_model.ConstrainedSketch(name='__profile__', sheetSize=1.0)
                SHEET_LENGTH_p, SHEET_WIDTH_p, SHEET_THICKNESS_p = 0.25, 0.1, 0.0001
                p1_p = (0.0, -SHEET_WIDTH_p / 2.0); p2_p = (SHEET_LENGTH_p, -SHEET_WIDTH_p / 2.0); p3_p = (SHEET_LENGTH_p, SHEET_WIDTH_p / 2.0); p4_p = (0.0, SHEET_WIDTH_p / 2.0)
                p_mid_bottom_p = (SHEET_LENGTH_p / 2.0, -SHEET_WIDTH_p / 2.0 - args.depth); p_mid_top_p = (SHEET_LENGTH_p / 2.0, SHEET_WIDTH_p / 2.0 + args.depth)
                s_p.Spline(points=(p2_p, p_mid_bottom_p, p1_p)); s_p.Line(point1=p1_p, point2=p4_p); s_p.Spline(points=(p4_p, p_mid_top_p, p3_p)); s_p.Line(point1=p3_p, point2=p2_p)
                p_p = post_model.Part(name='SheetPart_P', dimensionality=THREE_D, type=DEFORMABLE_BODY); p_p.BaseShell(sketch=s_p)
                material_p = post_model.Material(name='HyperelasticMat')
                material_p.Hyperelastic(materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, volumetricResponse=VOLUMETRIC_DATA, table=((3e6, 0.0), ))
                post_model.HomogeneousShellSection(name='SheetSection', material='HyperelasticMat', thickness=SHEET_THICKNESS_p)
                p_p.SectionAssignment(region=p_p.Set(faces=p_p.faces.getSequenceFromMask(mask=('[#1 ]',), ), name='AllFaces'), sectionName='SheetSection', offsetType=MIDDLE_SURFACE)
                p_p.seedPart(size=0.0025, deviationFactor=0.1, minSizeFactor=0.1)
                p_p.generateMesh()
                p_p.Set(edges=p_p.edges.getByBoundingBox(xMin=-0.001, xMax=0.001), name='Fix_Edges')
                p_p.Set(edges=p_p.edges.getByBoundingBox(xMin=SHEET_LENGTH_p-0.001, xMax=SHEET_LENGTH_p+0.001), name='Disp_Edges')
                a_p = post_model.rootAssembly; a_p.Instance(name='SheetInst_P', part=p_p, dependent=ON)
                post_model.StaticRiksStep(name='Riks', previous='Initial', nlgeom=ON, maxNumInc=200, initialArcInc=0.01)
                post_model.EncastreBC(name='FixLeft', createStepName='Initial', region=a_p.instances['SheetInst_P'].sets['Fix_Edges'])
                post_model.DisplacementBC(name='Riks_Disp', createStepName='Riks', region=a_p.instances['SheetInst_P'].sets['Disp_Edges'], u1=-0.005, u2=0, u3=0, ur1=0, ur2=0, ur3=0)
                
                temp_job_name = f"{job_name}_temp_inp"
                temp_job = mdb.Job(name=temp_job_name, model=post_model, type=ANALYSIS)
                temp_job.writeInput(consistencyChecking=OFF)
                
                original_inp_path = f"{temp_job_name}.inp"
                
                with open(original_inp_path, 'r') as f_in:
                    lines = f_in.readlines()
                
                with open(original_inp_path, 'w') as f_out:
                    for line in lines:
                        if line.strip().lower().startswith('*step'):
                            imperfection_string = f"*IMPERFECTION, FILE={buckle_job_name}, STEP=1\n1, {SHEET_THICKNESS_p * 0.01}\n"
                            f_out.write(imperfection_string)
                        f_out.write(line)

                command = f"abaqus job={job_name} input={original_inp_path} interactive"
                write_log(f"Executing command: {command}")
                process = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
                
                if process.returncode != 0:
                     write_log(f"Abaqus execution failed for {job_name}. Stderr: {process.stderr}")

                post_odb_path = f'{job_name}.odb'
                if os.path.exists(post_odb_path):
                    try:
                        odb_p = openOdb(path=post_odb_path, readOnly=True)
                        if 'Riks' in odb_p.steps and len(odb_p.steps['Riks'].frames) > 0:
                            # ### 최종 수정: 올바른 ODB API 사용 ###
                            last_frame = odb_p.steps['Riks'].frames[-1]
                            if 'U' in last_frame.fieldOutputs:
                                u_field = last_frame.fieldOutputs['U']
                                u3_values = [v.data[2] for v in u_field.values]
                                if u3_values:
                                    result_value = max(u3_values) - min(u3_values)
                            # ### 수정 종료 ###
                        odb_p.close()
                    except Exception as odb_e:
                        write_log(f"Post-buckle ODB access error for {job_name}: {odb_e}")
        
        with open(f"result_{args.job_name}.txt", 'w') as f:
            f.write(str(result_value))
        
        if result_value != "NaN":
            write_log(f"Success! Result: {result_value}")
        else:
            write_log("Analysis failed or returned no valid result.")

    except Exception as e:
        write_log(f"CATASTROPHIC ERROR: {e}\n{traceback.format_exc()}")
        with open(f"result_{args.job_name}.txt", 'w') as f:
            f.write("NaN")

if __name__ == '__main__':
    main()