clear; clc; close all;
% run_tf_simulation_hybrid.m
% メインスクリプト: MATLABで前処理を行い、Pythonで最適化を実行する
import python.*;
np = py.importlib.import_module('numpy');

% --- 0. Python環境の設定 (必要に応じて) ---
% pyenv('Version', 'C:/path/to/your/python.exe');
% Pythonスクリプトと同じディレクトリにいることを確認
if ~isfile('B_py_optimise_task.py')
    error('py_optimise_task.pyが見つかりません。MATLABの現在のディレクトリを確認してください。');
end

%% --- 1. シミュレーション設定 ---
fprintf('--- MATLAB: Setting up configuration ---\n');
cfg = struct();
%Load data
cfg.modelFile = '/n/home04/rfushio/Main/QPC-bridge.mph'; 
model = mphopen(cfg.modelFile);
cfg.rhoFile = 'B_rho_data.csv';
cfg.excFile = 'B_Exc_data_digitized.csv';
% Physics
cfg.e = 1.602e-19; cfg.epsilon_0 = 8.854e-12; cfg.h = 6.62607015e-34;
cfg.B = 13.0; cfg.dt = 30e-9; cfg.db = 30e-9;
cfg.epsilon_perp = 3.0; cfg.epsilon_parallel = 6.6;
% Grid
cfg.Nx = 32; cfg.Ny = 32; cfg.Lx = 400; cfg.Ly = 400;
cfg.x0  = linspace(-cfg.Lx/2, cfg.Lx/2, cfg.Nx); cfg.y0  = linspace(-cfg.Ly/2, cfg.Ly/2, cfg.Ny); cfg.z0=35;
[cfg.x_vec,cfg.y_vec,cfg.z]=meshgrid(cfg.x0,cfg.y0,cfg.z0);
cfg.coord = [cfg.x_vec(:),cfg.y_vec(:),cfg.z(:)]'; % ' has a meaning of "transpose"
cfg.Xmat = reshape(cfg.x_vec, [cfg.Nx,cfg.Ny]);
cfg.Ymat = reshape(cfg.y_vec, [cfg.Nx,cfg.Ny]);
cfg.Zmat = reshape(cfg.z, [cfg.Nx,cfg.Ny]);   
% Arbitrary Parameters
cfg.potential_scale = 1.0; cfg.potential_offset = 0.0;
% Pythonのbasinhopping用のパラメータ
cfg.niter = 5; % basinhoppingのイテレーション回数
cfg.step_size = 0.5;
cfg.lbfgs_maxiter = 1000;
cfg.lbfgs_maxfun = 200000;


%% --- 2.Prepare External Potential by Running COMSOL ---
fprintf('--- MATLAB: Preparing external potential ---\n');

% running the study %%
std = model.study("std1");
model.component('comp1').physics('es').feature('sfcd1').active(true);
% Interpolate the charge density from table"
f = model.func('int1').active(true);
f.set('source','table');
f.set('filename',fullfile(pwd,cfg.rhoFile));
f.set('argunit','nm');
f.set('fununit','C/m^2');
f.set('nargs',2);
f.set('extrap','linear');
model.component('comp1').physics('es').feature('sfcd1').set('rhoqs','int1(x,y)');
%run & save
mphrun(model,"std1")
mphsave(model)
% Extracting data by "mphinterp" %%
V = mphinterp(model, 'V','coord',cfg.coord);
Vmat = reshape(V * 1e-7, [cfg.Nx, cfg.Ny]);   % V_cgsa and reshape
% CSV
X = cfg.Xmat(:); Y = cfg.Ymat(:); Z = cfg.Zmat(:); Vout = Vmat(:);
pot_T = table(X, Y, Z, Vout, 'VariableNames', {'x_nm','y_nm','z_nm','V'});
writetable(pot_T, 'B_potential_data.csv');
% Extract potential
all_data = readtable('B_potential_data.csv');
V_vector = all_data.V;
Phi_grid_raw = reshape(V_vector, [cfg.Nx, cfg.Ny]); % 'might be needed
Phi_grid = Phi_grid_raw * cfg.potential_scale + cfg.potential_offset;

%% --- 3.Calling python optimisation function ---
fprintf('--- MATLAB: Calling Python for optimisation ---\n');
Phi_grid_py = np.array(Phi_grid);
% Pythonに渡す前に単位をメートルに変換
x_vec_m_py    = np.array(cfg.x_vec * 1e-9);
y_vec_m_py    = np.array(cfg.y_vec * 1e-9);
tic;
nu_smoothed_py = py.B_py_optimise_task.run_optimisation(py.dict(cfg), Phi_grid_py, x_vec_m_py, y_vec_m_py);
elapsed_time = toc;
fprintf('--- MATLAB: Python task completed in %.2f seconds ---\n', elapsed_time);


%% --- 4. Obtain the result and Display it ---
% Pythonから返されたNumPy配列をMATLABのdouble配列に変換
nu_smoothed = double(nu_smoothed_py);
nu = nu_smoothed(:);  % フラット化
rho = (nu*cfg.e*cfg.B)/cfg.h;
% テーブルにまとめてヘッダー付き CSV 出力
rho_T = table(X, Y, rho, 'VariableNames', {'x_nm','y_nm','nu'});
writetable(rho_T, 'B_rho_data.csv');

fprintf('----------------------------------------\n');

% 結果をプロット
figure;
imagesc(cfg.x0, cfg.y0, nu_smoothed');
axis image;
set(gca, 'YDir', 'normal');
title('Smoothed Filling Factor \nu(r) (calculated by Python)');
xlabel('x [nm]');
ylabel('y [nm]');
colorbar;
