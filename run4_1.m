function output_value = run4_1(alpha_val, th_W_ratio_val, fidelity_level_input, target_strain_percentage)

    output_value = NaN;
    import com.comsol.model.*
    import com.comsol.model.util.*

    model = []; 
    try
        model = ModelUtil.create('Model');
        model = setup_common_components(model, alpha_val, th_W_ratio_val);
        
        if fidelity_level_input == 0.0 
            
            model = setup_lf_study_and_solver(model); 
            
            fprintf('  Running LF (Buckling) analysis...\n');
            model.study('std_buckling').run(); 
            output_value = extract_lf_results(model); 
            
        elseif fidelity_level_input == 1.0 
            % HF는 좌굴 모드 정보가 필요하므로 LF 스터디가 선행되어야 합니다.
            model = setup_lf_study_and_solver(model);
            model = setup_hf_study_and_solver(model, target_strain_percentage);
            
            fprintf('  Running LF (for modes) then HF (Post-buckling) analysis...\n');
            model.study('std_buckling').run();
            model.study('std_postbuckling').run();
            
            output_value = extract_hf_results(model);
        else
            fprintf('Error: Invalid fidelity level input: %f\n', fidelity_level_input);
            output_value = NaN;
        end
        
    catch ME
        fprintf('ERROR during COMSOL execution for alpha=%.4f, th/W=%.6f\n', alpha_val, th_W_ratio_val);
        fprintf('Error message: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for k_err=1:length(ME.stack)
            fprintf('  File: %s, Name: %s, Line: %d\n', ME.stack(k_err).file, ME.stack(k_err).name, ME.stack(k_err).line);
        end
        output_value = NaN; 
    end
    
    if ~isempty(model)
        ModelUtil.remove('Model');
    end
end

function model = setup_common_components(model, alpha, th_w_ratio)
    % --- 파라미터 ---
    model.param.set('mu', '6[MPa]');
    model.param.set('W', '10[cm]');
    model.param.set('alpha', num2str(alpha));
    model.param.set('th', ['W*' num2str(th_w_ratio)]);
    model.param.set('L', 'alpha*W');
    model.param.set('numX', 'L/0.5[mm]'); % 메쉬 밀도 파라미터
    model.param.set('numY', 'W/1[mm]');
    model.param.set('geomImpFactor', '1E-4'); % 기하학적 결함 계수 (두께의 1/10000)

    comp1 = model.component.create('comp1', true);
    geom1 = comp1.geom.create('geom1', 3);
    wp1 = geom1.feature.create('wp1', 'WorkPlane');
    wp1.geom.create('r1', 'Rectangle').set('size', {'L' 'W'});

    geom1.run;

    % --- 경계 선택 (Box Selection) ---
    % mphselectcoords 대신 명시적인 선택 기능 사용
    L_val = model.param.evaluate('L');
    eps = 1e-6; % 좌표 비교를 위한 작은 허용 오차

    % 1. 왼쪽 경계를 선택하는 'Box' 셀렉션 생성
    sel_left = comp1.selection.create('sel_left_edge', 'Box');
    sel_left.set('entitydim', 1); % 경계를 선택 대상으로 지정
    sel_left.set('xmin', -L_val/2 - eps);
    sel_left.set('xmax', -L_val/2 + eps);

    % 2. 오른쪽 경계를 선택하는 'Box' 셀렉션 생성
    sel_right = comp1.selection.create('sel_right_edge', 'Box');
    sel_right.set('entitydim', 1); % 경계를 선택 대상으로 지정
    sel_right.set('xmin', L_val/2 - eps);
    sel_right.set('xmax', L_val/2 + eps);
    
    mat1 = comp1.material.create('mat1', 'Common');
    mat1.propertyGroup('def').set('density', '500'); % 가상 밀도
    mat1.propertyGroup.create('Lame', 'Lame');
    mat1.propertyGroup('Lame').set('muLame', 'mu');
    mat1.propertyGroup.create('shell', 'shell', 'Shell').set('lth', 'th');

    % --- 물리 법칙 ---
    shell = comp1.physics.create('shell', 'Shell', 'geom1');
    shell.create('lhmm1', 'LayeredHyperelasticModel',2);
    shell.feature('lhmm1').selection.all; % Layered Hyperelastic Material
    shell.feature('lhmm1').set('shelllist', 'none');
    shell.feature('lhmm1').set('MixedFormulationIncompressible', 'implicitIncompressibility');
    shell.feature('lhmm1').set('Compressibility_NeoHookean', 'Incompressible');
    
   % 3. 생성된 셀렉션을 사용하여 경계 조건 적용
    fix1 = shell.create('fix1', 'Fixed', 1);
    fix1.selection.named('sel_left_edge'); % 이름으로 셀렉션 지정

    pd1 = shell.create('pd1', 'Displacement1', 1);
    pd1.set('Direction', {'prescribed'; 'prescribed'; 'prescribed'});
    pd1.set('U0', {'nominalStrain*L'; '0'; '0'});
    pd1.selection.named('sel_right_edge'); % 이름으로 셀렉션 지정

    % --- 메쉬 ---
    mesh1 = comp1.mesh.create('mesh1');
    mesh1.create('map1', 'Map').selection.geom(geom1.tag, 2);
    mesh1.run;
end

function model = setup_lf_study_and_solver(model)
    study_lf = model.study.create('std_buckling');
    study_lf.create('stat', 'Stationary');
    study_lf.create('buckling', 'LinearBuckling');
    
    % [수정됨] 하중 방향을 인장(+)으로 설정
    model.component('comp1').physics('shell').feature('pd1').set('U0', {'L*0.01', '0', '0'});

    sol_lf = model.sol.create('sol_lf');
    sol_lf.attach('std_buckling');

    % 성공했던 코드 기반의 수동 솔버 시퀀스 사용
    sol_lf.create('st1', 'StudyStep');
    sol_lf.create('v1', 'Variables');
    sol_lf.create('s1', 'Stationary');
    sol_lf.create('st2', 'StudyStep');
    sol_lf.create('v2', 'Variables');
    sol_lf.create('e1', 'Eigenvalue');

    sol_lf.feature('s1').feature.create('fc1', 'FullyCoupled');
    sol_lf.feature('s1').feature.remove('fcDef');

    sol_lf.feature('st2').set('studystep', 'buckling');
    sol_lf.feature('v2').set('initmethod', 'sol');
    sol_lf.feature('v2').set('initsol', 'sol_lf');
    sol_lf.feature('v2').set('notsolmethod', 'sol');
    sol_lf.feature('v2').set('notsol', 'sol_lf');
    
    eig_solver = sol_lf.feature('e1');
    eig_solver.set('control', 'buckling');
    eig_solver.set('neigs', 10);
    
    % [수정됨] 인장 좌굴을 찾기 위한 솔버 설정
    eig_solver.set('shift', '0');      % 크기가 가장 작은 고유치를 찾도록 shift=0으로 설정
    eig_solver.set('eigwhich', 'sr');  % 'smallest magnitude' 옵션으로 변경
end

function model = setup_hf_study_and_solver(model, target_strain_percentage)
    comp1 = model.component('comp1');
    imp = comp1.common.create('bcki1', 'BucklingImperfection');
    imp.set('Study', 'std_buckling'); 
    imp.set('ModesScales', {'1', 'th*geomImpFactor'});

    pres_def = comp1.common.create('pres_shell', 'PrescribedDeformationDeformedGeometry');
    pres_def.selection.all;
    pres_def.set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});

    study_hf = model.study.create('std_postbuckling');
    stat_hf = study_hf.create('stat1', 'Stationary');
    stat_hf.set('nlgeom', 'on');
    
    stat_hf.set('useparam', true);
    stat_hf.set('pname', {'nominalStrain'});
    strain_frac = target_strain_percentage / 100;
    plist_str = sprintf('range(0, %g, %g)', strain_frac/25, strain_frac);
    stat_hf.set('plistarr', {plist_str});
    
    % [수정됨] 하중 방향을 인장(+)으로 설정
    comp1.physics('shell').feature('pd1').set('u', {'L*nominalStrain', '0', '0'});

    sol_hf = model.sol.create('sol_hf');
    sol_hf.study('std_postbuckling');
    sol_hf.createAutoSequence('std_postbuckling');
end

function lambda_val = extract_lf_results(model)
    frpintf("stop");
     lambda_val = NaN;
    try
        % 성공했던 코드는 두 번째 솔루션 데이터셋을 사용했으므로 'dset3'일 확률이 높음
        % COMSOL은 지오메트리(dset1), 정적해(dset2), 고유치해(dset3) 순으로 생성
        lambdas = mphglobal(model, 'lambda', 'dataset', 'dset3');
        valid_lambdas = lambdas(isreal(lambdas) & lambdas > 1e-6);
        if isempty(valid_lambdas)
            fprintf('  Warning: No valid positive real lambda values found in dset3.\n');
        else
            lambda_val = min(valid_lambdas);
        end
    catch ME_extract
        fprintf('  Error extracting LF (lambda) from dset3: %s\n', ME_extract.message);
    end
end

function amplitude_val = extract_hf_results(model)
    amplitude_val = NaN;
    try
        % 후좌굴 해석 결과는 보통 마지막 데이터셋(dset4)에 저장됨
        max_amp_expr = 'max(0, maxop1(shell.w)-minop1(shell.w))'; % max(0,...)으로 음수 진폭 방지
        amplitudes = mphglobal(model, max_amp_expr, 'dataset', 'dset4', 'outersolnum', 'all');
        
        valid_amplitudes = amplitudes(isreal(amplitudes) & ~isnan(amplitudes));
        if isempty(valid_amplitudes)
            amplitude_val = 0;
        else
            amplitude_val = max(valid_amplitudes);
        end
    catch ME_extract
        fprintf('  Error extracting HF (amplitude) result: %s\n', ME_extract.message);
    end
end