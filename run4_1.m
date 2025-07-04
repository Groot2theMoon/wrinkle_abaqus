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
    model.param.set('numX', 'L/1[mm]'); % 메쉬 밀도 파라미터
    model.param.set('numY', 'W/2[mm]');
    model.param.set('geomImpFactor', '1E-4'); % 기하학적 결함 계수 (두께의 1/10000)

    comp1 = model.component.create('comp1', true);
    geom1 = comp1.geom.create('geom1', 2);
    wp1 = geom1.feature.create('wp1', 'WorkPlane');
    wp1.geom.create('r1', 'Rectangle').set('size', {'L' 'W'});
    geom1.run;
    
    mat1 = comp1.material.create('mat1', 'Common');
    mat1.propertyGroup('def').set('density', '500'); % 가상 밀도
    mat1.propertyGroup.create('Lame', 'Lame');
    mat1.propertyGroup('Lame').set('muLame', 'mu');
    mat1.propertyGroup.create('shell', 'shell', 'Shell').set('lth', 'th');

    % --- 물리 법칙 ---
    shell = comp1.physics.create('shell', 'Shell', 'geom1');
    shell.feature('lhmm1').selection.all; % Layered Hyperelastic Material
    
    % --- 경계 선택 ---
    bnd_selector = @(x, y, z) ['box', num2str(x(1)), num2str(x(2)), num2str(y(1)), num2str(y(2)), num2str(z(1)), num2str(z(2))];
    left_edge_selector = geom1.selection.create('left_edge', 'Boundary');
    left_edge_selector.set(shell.allEdge, 'point-in-box', bnd_selector([-inf, -0.49*L], [-inf, inf], [-inf, inf]));
    
    right_edge_selector = geom1.selection.create('right_edge', 'Boundary');
    right_edge_selector.set(shell.allEdge, 'point-in-box', bnd_selector([0.49*L, inf], [-inf, inf], [-inf, inf]));

    shell.create('fix1', 'Fixed', 1).selection.named('left_edge');
    shell.create('pd1', 'PrescribedDisplacement', 1).selection.named('right_edge');

    % --- 메쉬 ---
    mesh1 = comp1.mesh.create('mesh1');
    mesh1.create('map1', 'Map').selection.geom(geom1.tag, 2);
    mesh1.run;
end

function model = setup_lf_study_and_solver(model)
    study_lf = model.study.create('std_buckling');
    study_lf.create('stat', 'Stationary');
    study_lf.create('buckling', 'LinearBuckling');
    
    % 단위 변위를 가하여 응력 상태 생성
    model.component('comp1').physics('shell').feature('pd1').set('u', {'L*0.01', '0', '0'});
    
    sol_lf = model.sol.create('sol_lf');
    sol_lf.study('std_buckling');
    sol_lf.createAutoSequence('std_buckling');
    
    eig_solver = sol_lf.feature('e1');
    eig_solver.set('neigs', 5); % 5개의 좌굴 모드 계산
    eig_solver.set('shift', '1'); % 가장 작은 양의 고유치를 찾기 위함
end

function model = setup_hf_study_and_solver(model, target_strain_percentage)
    % Imperfection : LF 스터디(std_buckling)의 결과를 초기 결함으로 사용
    imp = model.component('comp1').common.create('bcki1', 'BucklingImperfection');
    imp.set('Study', 'std_buckling'); % LF 스터디 참조
    imp.set('ModesScales', {'1', 'th*geomImpFactor'}); % 첫번째 모드를 결함 형상으로 사용
    
    pres_def = model.component('comp1').common.create('pres_shell', 'PrescribedDeformationDeformedGeometry');
    pres_def.selection.all;
    pres_def.set('prescribedDeformation', {'bcki1.dshellX' 'bcki1.dshellY' 'bcki1.dshellZ'});

    study_hf = model.study.create('std_postbuckling');
    stat_hf = study_hf.create('stat1', 'Stationary');
    stat_hf.set('nlgeom', 'on'); % 기하학적 비선형성
    
    % 파라메트릭 스윕 설정: Strain을 0에서 목표치까지 증가
    stat_hf.set('useparam', true);
    stat_hf.set('pname', {'nominalStrain'});
    strain_frac = target_strain_percentage / 100;
    plist_str = sprintf('range(0, %g, %g)', strain_frac/25, strain_frac);
    stat_hf.set('plistarr', {plist_str});
    
    % --- 물리 설정 ---
    % 변위 조건을 파라미터 'nominalStrain'에 연동
    model.component('comp1').physics('shell').feature('pd1').set('u', {'L*nominalStrain', '0', '0'});

    % --- 솔버 ---
    sol_hf = model.sol.create('sol_hf');
    sol_hf.study('std_postbuckling');
    sol_hf.createAutoSequence('std_postbuckling');
    
    % Riks 메서드
    % 예시: sol_hf.feature('s1').set('continuation', 'Riks');
end

function lambda_val = extract_lf_results(model)
    try
        lambdas = mphglobal(model, 'lambda', 'dataset', 'dset2');
        valid_lambdas = lambdas(isreal(lambdas) & lambdas > 1e-6);
        if isempty(valid_lambdas)
            fprintf('  Warning: No valid positive real lambda values found.\n');
            lambda_val = NaN;
        else
            lambda_val = min(valid_lambdas);
        end
    catch ME_extract
        fprintf('  Error extracting LF (lambda) result: %s\n', ME_extract.message);
        lambda_val = NaN;
    end
end

function amplitude_val = extract_hf_results(model)
    try
        max_amp_expr = '(maxop1(w)-minop1(w))';
        amplitudes = mphglobal(model, max_amp_expr, 'dataset', 'dset4', 'outersolnum', 'all');
        
        valid_amplitudes = amplitudes(isreal(amplitudes) & ~isnan(amplitudes));
        if isempty(valid_amplitudes)
            fprintf('  Warning: No valid wrinkle amplitudes found.\n');
            amplitude_val = 0; % 주름이 생기지 않았다고 간주
        else
            amplitude_val = max(valid_amplitudes);
        end
    catch ME_extract
        fprintf('  Error extracting HF (amplitude) result: %s\n', ME_extract.message);
        amplitude_val = NaN;
    end
end