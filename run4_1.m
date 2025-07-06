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
    model.param.set('geomImpFactor', '1E-4');
    model.param.set('nominalStrain', '0.0');

    % [핵심 수정] 컴포넌트는 반드시 3D여야 함
    comp1 = model.component.create('comp1', true);
    geom1 = comp1.geom.create('geom1', 3); 
    
    % run4와 같이 WorkPlane 사용
    wp1 = geom1.feature.create('wp1', 'WorkPlane');
    wp1.set('unite', true);
    wp1.geom.create('r1', 'Rectangle').set('size', {'L' 'W'});
    geom1.run;

    % --- 안정적인 Box 경계 선택 (좌표계 주의) ---
    % WorkPlane의 기본 위치는 z=0 입니다.
    L_val = model.param.evaluate('L');
    W_val = model.param.evaluate('W');
    eps = 1e-6; 
    
    % 왼쪽 경계: x=0
    sel_left = comp1.selection.create('sel_left_edge', 'Box');
    sel_left.set('entitydim', 1);
    sel_left.set('xmin', -eps); sel_left.set('xmax', eps);
    sel_left.set('ymin', -eps); sel_left.set('ymax', W_val + eps); % y 범위 지정

    % 오른쪽 경계: x=L
    sel_right = comp1.selection.create('sel_right_edge', 'Box');
    sel_right.set('entitydim', 1);
    sel_right.set('xmin', L_val - eps); sel_right.set('xmax', L_val + eps);
    sel_right.set('ymin', -eps); sel_right.set('ymax', W_val + eps); % y 범위 지정

    % --- 재료 ---
    mat1 = comp1.material.create('mat1', 'Common');
    mat1.propertyGroup('def').set('density', '500');
    mat1.propertyGroup.create('Lame', 'Lame').set('muLame', 'mu');
    mat1.propertyGroup.create('shell', 'shell', 'Shell').set('lth', 'th');

    % --- 물리 법칙 ---
    shell = comp1.physics.create('shell', 'Shell', 'geom1');
    shell.create('lhmm1', 'LayeredHyperelasticModel', 2);
    shell.feature('lhmm1').selection.all;
    shell.feature('lhmm1').set('MixedFormulationIncompressible', 'implicitIncompressibility');
    
    % run4와 유사한 간단한 경계 조건 사용
    fix1 = shell.create('fix1', 'Fixed', 1);
    fix1.selection.named('sel_left_edge'); 

    pd1 = shell.create('pd1', 'Displacement1', 1);
    pd1.selection.named('sel_right_edge');
    
    % --- 메쉬 ---
    mesh1 = comp1.mesh.create('mesh1');
    mesh1.create('map1', 'Map');
    mesh1.run;
end

function model = setup_lf_study_and_solver(model)
    study_lf = model.study.create('std_buckling');
    study_lf.create('stat', 'Stationary');
    study_lf.create('buckling', 'LinearBuckling');
    
    % --- [핵심 수정] 'Displacement1' 기능에 맞는 속성 설정 ---
    % 'pd1' 기능의 속성을 직접 설정합니다.
    pd1 = model.component('comp1').physics('shell').feature('pd1');
    
    % 'Displacement1'은 방향과 변위 벡터(U0)로 제어됩니다.
    pd1.set('Direction', {'prescribed'; 'prescribed'; 'prescribed'});
    pd1.set('U0', {'L*0.01'; '0'; '0'}); % 인장 하중 설정

    sol_lf = model.sol.create('sol_lf');
    sol_lf.attach('std_buckling');
    
    % --- run4.m의 성공적인 솔버 시퀀스를 완벽하게 복제 ---
    sol_lf.create('st1', 'StudyStep');
    sol_lf.create('v1', 'Variables');
    sol_lf.create('s1', 'Stationary');
    su1 = sol_lf.create('su1', 'StoreSolution');
    sol_lf.create('st2', 'StudyStep');
    sol_lf.create('v2', 'Variables');
    sol_lf.create('e1', 'Eigenvalue');

    sol_lf.feature('s1').feature.create('fc1', 'FullyCoupled');
    sol_lf.feature('s1').feature.remove('fcDef');
    
    sol_lf.feature('st2').set('studystep', 'buckling');
    
    v2 = sol_lf.feature('v2');
    v2.set('initmethod', 'sol');
    v2.set('initsol', 'sol_lf');
    v2.set('initsoluse', 'sol3'); % su1이 생성하는 해 이름
    
    e1 = sol_lf.feature('e1');
    e1.set('control', 'buckling');
    e1.set('neigs', 10);
    e1.set('linpmethod', 'sol');
    e1.set('linpsol', 'sol_lf');
    e1.set('linpsoluse', 'sol3'); % su1이 생성하는 해 이름
    e1.set('shift', '0');
    e1.set('eigwhich', 'sr');
end

function model = setup_hf_study_and_solver(model, target_strain_percentage)
    % ... (이전의 초기 결함 설정 코드는 동일)
    comp1 = model.component('comp1');
    imp = comp1.common.create('bcki1', 'BucklingImperfection');
    imp.set('Study', 'std_buckling'); 
    imp.set('ModesScales', {'1', 'th*geomImpFactor'});
    pres_def = comp1.common.create('pres_shell', 'PrescribedDeformationDeformedGeometry');
    pres_def.selection.all;
    pres_def.set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});
    % ...

    study_hf = model.study.create('std_postbuckling');
    stat_hf = study_hf.create('stat', 'Stationary');
    stat_hf.set('nlgeom', 'on');
    
    stat_hf.set('useparam', true);
    stat_hf.set('pname', {'nominalStrain'});
    strain_frac = target_strain_percentage / 100;
    plist_str = sprintf('range(0, %g, %g)', strain_frac/25, strain_frac);
    stat_hf.set('plistarr', {plist_str});
    
    comp1.physics('shell').feature('pd1').set('U0', {'L*nominalStrain', '0', '0'});

    sol_hf = model.sol.create('sol_hf');
    sol_hf.study('std_postbuckling');
    
    % --- [핵심 추가] HF 결과를 저장할 dset4를 미리 생성 ---
    dset4 = model.result.dataset.create('dset4', 'Solution');
    dset4.set('solsel', 'sol_hf');
    dset4.set('solutions', {'sol_hf'});
    
    sol_hf.createAutoSequence('std_postbuckling');
end

% 이전에 사용했던 강력한 진단 버전의 extract_lf_results 함수를 그대로 사용하세요.
% (바로 이전 답변에 있던 코드)
function lambda_val = extract_lf_results(model)
    lambda_val = NaN;
    try
        fprintf('--- Inspecting available datasets in the model ---\n');
        dset_tags_java = model.result.dataset.tags; 
        dset_tags = cell(dset_tags_java); 
        fprintf('  Available dataset tags: %s\n', strjoin(dset_tags, ', '));
        fprintf('--------------------------------------------------\n');
        
        dsets_to_try = flip(dset_tags);
        lambda_found = false;
        for i = 1:length(dsets_to_try)
            current_dset = dsets_to_try{i};
            if strcmp(current_dset, 'dset1'), continue; end
            fprintf('  Attempting to extract lambda from dataset: ''%s''\n', current_dset);
            try
                lambdas = mphglobal(model, 'lambda', 'dataset', current_dset);
                valid_lambdas = lambdas(isreal(lambdas) & lambdas > 1e-6);
                if ~isempty(valid_lambdas)
                    lambda_val = min(valid_lambdas);
                    fprintf('  >>> SUCCESS! Found lambda = %f in dataset ''%s''.\n', lambda_val, current_dset);
                    lambda_found = true;
                    break;
                else
                    fprintf('    ...No valid lambda values found.\n');
                end
            catch
                fprintf('    ...Could not extract ''lambda''.\n');
            end
        end
        if ~lambda_found
            fprintf('  Warning: Failed to find lambda.\n');
        end
    catch ME_extract
        fprintf('  An unexpected error occurred: %s\n', ME_extract.message);
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