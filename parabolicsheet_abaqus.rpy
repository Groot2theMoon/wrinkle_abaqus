# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2025 replay file
# Internal Version: 2024_09_20-22.00.46 RELr427 198590
# Run by user on Thu Jul 10 13:22:57 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=204.628112792969, 
    height=243.911117553711)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
Mdb()
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=1.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.Line(point1=(0.0, 0.0), point2=(0.0, 0.1))
s.VerticalConstraint(entity=g[2], addUndoState=False)
s.Line(point1=(0.25, 0.0), point2=(0.25, 0.1))
s.VerticalConstraint(entity=g[3], addUndoState=False)
s.Spline(points=((0.0, 0.0), (0.125, 0.01), (0.25, 0.0)))
s.undo()
s.Spline(points=((0.0, 0.0), (0.125, -0.01), (0.25, 0.0)))
s.Spline(points=((0.0, 0.1), (0.125, 0.11), (0.25, 0.1)))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Density(table=((1000.0, ), ))
mdb.models['Model-1'].materials['Material-1'].Hyperelastic(
    materialType=ISOTROPIC, testData=OFF, type=NEO_HOOKE, 
    volumetricResponse=VOLUMETRIC_DATA, table=((3000000.0, 0.0), ))
mdb.models['Model-1'].HomogeneousShellSection(name='Section-1', 
    preIntegrate=OFF, material='Material-1', thicknessType=UNIFORM, 
    thickness=0.0001, thicknessField='', nodalThicknessField='', 
    idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
    thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
    integrationRule=SIMPSON, numIntPts=5)
p = mdb.models['Model-1'].parts['Part-1']
f = p.faces
faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
region = p.Set(faces=faces, name='Set-1')
p = mdb.models['Model-1'].parts['Part-1']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', 
    timePeriod=10.0, maxNumInc=1000, initialInc=1.0, minInc=1e-06, maxInc=1.0, 
    nlgeom=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['Model-1'].BuckleStep(name='Step-2', previous='Step-1', numEigen=3, 
    maxEigen=100.0, eigensolver=LANCZOS, minEigen=0.0, blockSize=DEFAULT, 
    maxBlocks=DEFAULT)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-2')
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#8 ]', ), )
region = a.Set(edges=edges1, name='Set-1')
mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Initial', 
    region=region, localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
a = mdb.models['Model-1'].rootAssembly
e1 = a.instances['Part-1-1'].edges
edges1 = e1.getSequenceFromMask(mask=('[#2 ]', ), )
region = a.Set(edges=edges1, name='Set-2')
mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    region=region, u1=2.5e-05, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-2')
mdb.models['Model-1'].boundaryConditions['BC-2'].setValuesInStep(
    stepName='Step-2', u1=0.0025, buckleCase=PERTURBATION_AND_BUCKLING)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
    bcs=OFF, predefinedFields=OFF, connectors=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF, mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
p = mdb.models['Model-1'].parts['Part-1']
e = p.edges
pickedEdges = e.getSequenceFromMask(mask=('[#a ]', ), )
p.seedEdgeByNumber(edges=pickedEdges, number=100, constraint=FINER)
p = mdb.models['Model-1'].parts['Part-1']
p.generateMesh()
a1 = mdb.models['Model-1'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
    meshTechnique=OFF)
mdb.Job(name='Pre-Strain-Buckle', model='Model-1', description='', 
    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, numCpus=1, numGPUs=0)
mdb.jobs['Pre-Strain-Buckle'].submit(consistencyChecking=OFF)
#: The job input file "Pre-Strain-Buckle.inp" has been submitted for analysis.
#: Job Pre-Strain-Buckle: Analysis Input File Processor completed successfully.
#: Job Pre-Strain-Buckle: Abaqus/Standard completed successfully.
#: Job Pre-Strain-Buckle completed successfully. 
o3 = session.openOdb(
    name='C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Pre-Strain-Buckle.odb')
#: Model: C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Pre-Strain-Buckle.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          4
#: Number of Steps:              2
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.625654, 
    farPlane=1.063, width=0.382599, height=0.168685, viewOffsetX=-0.0115355, 
    viewOffsetY=0.0117641)
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
#: The contents of viewport "Viewport: 1" have been copied to the clipboard.
mdb.Model(name='Model-1-Copy', objectToCopy=mdb.models['Model-1'])
#: The model "Model-1-Copy" has been created.
a = mdb.models['Model-1-Copy'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
del mdb.models['Model-1-Copy'].steps['Step-2']
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.models['Model-1-Copy'].StaticRiksStep(name='Step-1', previous='Initial', 
    maintainAttributes=True, maxNumInc=10000, maxArcInc=1.0, nlgeom=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
p1 = mdb.models['Model-1-Copy'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
p = mdb.models['Model-1-Copy'].parts['Part-1']
n = p.nodes
nodes = n.getSequenceFromMask(mask=('[#0:795 #800 ]', ), )
p.Set(nodes=nodes, name='Set-2')
#: The set 'Set-2' has been created (1 node).
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
a1 = mdb.models['Model-1-Copy'].rootAssembly
a1.regenerate()
a = mdb.models['Model-1-Copy'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
regionDef=mdb.models['Model-1-Copy'].rootAssembly.allInstances['Part-1-1'].sets['Set-2']
mdb.models['Model-1-Copy'].steps['Step-1'].setValues(nodeOn=ON, 
    maximumDisplacement=0.25, region=regionDef, dof=1)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
mdb.models['Model-1-Copy'].boundaryConditions['BC-2'].setValues(u1=0.0025)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF)
import job
mdb.models['Model-1-Copy'].keywordBlock.synchVersions(
    storeNodesAndElements=False)
mdb.models['Model-1-Copy'].keywordBlock.replace(27, """
** ----------------------------------------------------------------
**
*Imperfection, file=Pre-Strain-Buckle, step=2
1, 0.01
**

** STEP: Step-1
**""")
mdb.jobs['Pre-Strain-Buckle'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "Pre-Strain-Buckle.inp".
mdb.Job(name='Post-Buckling', model='Model-1-Copy', description='', 
    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
    memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, numCpus=1, numGPUs=0)
mdb.jobs['Post-Buckling'].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "Post-Buckling.inp".
mdb.jobs['Post-Buckling'].submit(consistencyChecking=OFF, datacheckJob=True)
#: The job input file "Post-Buckling.inp" has been submitted for analysis.
#: Job Post-Buckling: Analysis Input File Processor completed successfully.
#: Job Post-Buckling: Abaqus/Standard completed successfully.
#: Job Post-Buckling completed successfully. 
mdb.jobs['Post-Buckling'].submit(consistencyChecking=OFF)
#: The job input file "Post-Buckling.inp" has been submitted for analysis.
#: Job Post-Buckling: Analysis Input File Processor completed successfully.
#: Error in job Post-Buckling: Too many attempts made for this increment
#: Error in job Post-Buckling: THE ANALYSIS HAS BEEN TERMINATED DUE TO PREVIOUS ERRORS. ALL OUTPUT REQUESTS HAVE BEEN WRITTEN FOR THE LAST CONVERGED INCREMENT.
#: Job Post-Buckling: Abaqus/Standard aborted due to errors.
#: Error in job Post-Buckling: Abaqus/Standard Analysis exited with an error - Please see the  message file for possible error messages if the file exists.
#: Job Post-Buckling aborted due to errors.
session.viewports['Viewport: 1'].setValues(
    displayedObject=session.odbs['C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Pre-Strain-Buckle.odb'])
o3 = session.openOdb(
    name='C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Post-Buckling.odb')
#: Model: C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Post-Buckling.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=84)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=17)
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(0.134861, 
    0.0455111, 0.592989), cameraUpVector=(0, 1, 0))
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=89)
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(0.733204, 
    0.0455111, -0.00535434))
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(0.134861, 
    0.0455111, 0.592989))
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=SCALE_FACTOR)
session.viewports['Viewport: 1'].animationController.play(duration=UNLIMITED)
session.animationOptions.setValues(frameRate=33)
session.viewports['Viewport: 1'].animationController.setValues(
    animationType=NONE)
a = mdb.models['Model-1-Copy'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=OFF, bcs=OFF, 
    predefinedFields=OFF, connectors=OFF, optimizationTasks=ON, 
    geometricRestrictions=ON, stopConditions=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=OFF, 
    constraints=OFF, connectors=OFF, engineeringFeatures=OFF, 
    adaptiveMeshConstraints=ON)
mdb.models['Model-1-Copy'].steps['Step-1'].setValues(maxNumInc=100000, 
    minArcInc=1e-09)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    adaptiveMeshConstraints=OFF)
mdb.jobs['Post-Buckling'].submit(consistencyChecking=OFF)
#: The job input file "Post-Buckling.inp" has been submitted for analysis.
#: Job Post-Buckling: Analysis Input File Processor completed successfully.
#: Error in job Post-Buckling: Too many attempts made for this increment
#: Error in job Post-Buckling: THE ANALYSIS HAS BEEN TERMINATED DUE TO PREVIOUS ERRORS. ALL OUTPUT REQUESTS HAVE BEEN WRITTEN FOR THE LAST CONVERGED INCREMENT.
#: Job Post-Buckling: Abaqus/Standard aborted due to errors.
#: Error in job Post-Buckling: Abaqus/Standard Analysis exited with an error - Please see the  message file for possible error messages if the file exists.
#: Job Post-Buckling aborted due to errors.
o3 = session.openOdb(
    name='C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Post-Buckling.odb')
#: Model: C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/Post-Buckling.odb
#: Number of Assemblies:         1
#: Number of Assembly instances: 0
#: Number of Part instances:     1
#: Number of Meshes:             1
#: Number of Element Sets:       4
#: Number of Node Sets:          5
#: Number of Steps:              1
session.viewports['Viewport: 1'].setValues(displayedObject=o3)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
    'Magnitude'), )
session.viewports['Viewport: 1'].odbDisplay.display.setValues(
    plotState=CONTOURS_ON_DEF)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U1'), )
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=89)
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U2'), )
session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
    variableLabel='U', outputPosition=NODAL, refinement=(COMPONENT, 'U3'), )
session.viewports['Viewport: 1'].view.setValues(cameraPosition=(0.134861, 
    0.0455111, 0.592989), cameraUpVector=(0, 1, 0))
session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(
    visibleEdges=NONE)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.558988, 
    farPlane=0.626909, width=0.393589, height=0.17353, 
    viewOffsetX=-0.000494634, viewOffsetY=0.000979785)
session.viewports[session.currentViewportName].odbDisplay.setFrame(
    step='Step-1', frame=89)
session.viewports['Viewport: 1'].view.setValues(width=0.39717, height=0.175109, 
    cameraPosition=(0.727285, 0.0465504, 4.03525e-05), cameraTarget=(0.134336, 
    0.0465504, 4.03525e-05), viewOffsetX=0, viewOffsetY=0)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.461417, 
    farPlane=0.741094, width=0.187138, height=0.0825076, 
    viewOffsetX=-0.00173303, viewOffsetY=-0.000474638)
session.viewports['Viewport: 1'].view.setValues(width=0.184553, 
    height=0.0813677, cameraPosition=(0.126029, 0.0459319, 0.603554), 
    cameraTarget=(0.126029, 0.0459319, 0.0022986), viewOffsetX=0, 
    viewOffsetY=0)
session.viewports['Viewport: 1'].view.setValues(nearPlane=0.580537, 
    farPlane=0.626491, width=0.297956, height=0.131366, 
    viewOffsetX=-0.000439421, viewOffsetY=-0.00169302)
mdb.saveAs(
    pathName='C:/Users/user/Desktop/Daeyoung/Platebuckling/0710_WideParabolicSheet/ParabolicSheet')
#: The model database has been saved to "C:\Users\user\Desktop\Daeyoung\Platebuckling\0710_WideParabolicSheet\ParabolicSheet.cae".
