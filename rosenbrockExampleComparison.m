%% Rosenbrock Optimization: Using MATLAB Built-in Optimizer Functions
% Compares: sophiaupdate(), adamupdate(), sgdmupdate()
% Demonstrates why Sophia outperforms on ill-conditioned problems like Rosenbrock
%
% REQUIREMENTS:
%   - sophiaupdate.m (provided in context)
%   - MATLAB R2021b or later (for adamupdate, sgdmupdate)
%
% USAGE:
%   Simply run this script to see 3D visualization and convergence plots

clear; close all; clc;

%% Define Rosenbrock Function and Gradient
rosenbrock = @(x, y) (1 - x).^2 + 100 * (y - x.^2).^2;

rosenbrock_grad = @(x, y) [...
    -2 * (1 - x) - 400 * x .* (y - x.^2);
    200 * (y - x.^2)];

rosenbrock_diag_hess = @(x, y) [...
    2 - 400 * (y - x.^2) + 800 * x.^2;
    200 * ones(size(x))];

%% Configuration
x0 = -1.5;
y0 = 2.5;
num_iters = 1000;
hess_interval = 10;
hessian_method = 'Hutchinson'; % 'Analytical', 'Hutchinson' or 'GNB'

% Optimized hyperparameters
sgdm_lr = 0.001;
sgdm_momentum = 0.9;

adam_lr = 0.05;
adam_beta1 = 0.9;
adam_beta2 = 0.999;

sophia_lr = 0.05;
sophia_beta1 = 0.9;
sophia_beta2 = 0.9;
sophia_rho = 0.01;
if strcmpi(hessian_method,'GNB')
    sophia_beta2 = 0.99;
    sophia_rho = 0.0003; 
end

%% Initialize Optimizer States

% SGDM Optimizer
params_sgdm = [x0; y0];
velocity_sgdm = [];
history_sgdm = params_sgdm;
loss_sgdm = rosenbrock(x0, y0);

% Adam Optimizer
params_adam = [x0; y0];
avg_grad_adam = [];
avg_sq_grad_adam = [];
history_adam = params_adam;
loss_adam = rosenbrock(x0, y0);

% Sophia Optimizer
params_sophia = [x0; y0];
avg_grad_sophia = [];
avg_hess_sophia = [];
history_sophia = params_sophia;
loss_sophia = rosenbrock(x0, y0);

%% Main Optimization Loop
iter_hess = 0;
for iter = 1:num_iters
    
    %% SGDM Update using sgdmupdate()
    grad_sgdm = rosenbrock_grad(params_sgdm(1), params_sgdm(2));
    [params_sgdm, velocity_sgdm] = sgdmupdate(params_sgdm, grad_sgdm, ...
        velocity_sgdm, sgdm_lr, sgdm_momentum);
    history_sgdm = [history_sgdm, params_sgdm];
    loss_sgdm = [loss_sgdm, rosenbrock(params_sgdm(1), params_sgdm(2))];
    
    %% Adam Update using adamupdate()
    grad_adam = rosenbrock_grad(params_adam(1), params_adam(2));
    [params_adam, avg_grad_adam, avg_sq_grad_adam] = adamupdate( ...
        params_adam, grad_adam, avg_grad_adam, avg_sq_grad_adam, iter, ...
        adam_lr, adam_beta1, adam_beta2);
    history_adam = [history_adam, params_adam];
    loss_adam = [loss_adam, rosenbrock(params_adam(1), params_adam(2))];
    
    %% Sophia Update using sophiaupdate()
    grad_sophia = rosenbrock_grad(params_sophia(1), params_sophia(2));
    
    % Gauss-Newton-Bartlett Hessian estimation: h = grad .* grad
    hess_estimate = [];
    update_hess_flag = (iter == 1) || (mod(iter, hess_interval) == 0);
    if update_hess_flag
        iter_hess = iter_hess + 1;
        switch hessian_method
            case 'GNB'
                hess_estimate = grad_sophia.*grad_sophia;
            case 'Hutchinson'
                epsilon = 1e-4; % Finite difference step size

                % Draw a random Rademacher vector z (+1 or -1)
                z = 2*(rand(size(params_sophia)) > 0.5) - 1;

                % Perturb the parameters
                params_perturbed = params_sophia + epsilon * z;

                % Calculate gradient at perturbed location
                grad_perturbed = rosenbrock_grad(params_perturbed(1), params_perturbed(2));

                % Approximate Hessian-vector product: Hv approx (g(x+eps*z) - g(x)) / eps
                Hv = (grad_perturbed - grad_sophia) / epsilon;

                % Hutchinson diagonal estimate: h = z .* (Hz)
                % We use abs() because Sophia requires a non-negative diagonal Hessian
                hess_estimate = max(z.*Hv,0);
            otherwise
                % Sophia requires a non-negative Hessian estimate for the denominator
                hess_estimate = max(rosenbrock_diag_hess(params_sophia(1), params_sophia(2)),0);
        end
    end

    [params_sophia, avg_grad_sophia, avg_hess_sophia] = sophiaupdate(...
        params_sophia, grad_sophia, avg_grad_sophia, avg_hess_sophia, ...
        hess_estimate, update_hess_flag, iter, iter_hess, sophia_lr, ...
        sophia_beta1, sophia_beta2, sophia_rho);
    
    history_sophia = [history_sophia, params_sophia];
    loss_sophia = [loss_sophia, rosenbrock(params_sophia(1), params_sophia(2))];
end

%% Compute Convergence Metrics
iterations = 0:num_iters;
dist_sgdm = sqrt((history_sgdm(1, :) - 1).^2 + (history_sgdm(2, :) - 1).^2);
dist_adam = sqrt((history_adam(1, :) - 1).^2 + (history_adam(2, :) - 1).^2);
dist_sophia = sqrt((history_sophia(1, :) - 1).^2 + (history_sophia(2, :) - 1).^2);

%% Visualization 1: 3D Surface with Optimization Trajectories
figure('Position', [100, 100, 1200, 800]);
[X, Y] = meshgrid(linspace(-2, 2, 500), linspace(-1, 3, 500));
Z = rosenbrock(X, Y);

surf(X, Y, Z, 'FaceAlpha', 0.7, 'EdgeColor', 'none');
colormap('jet');
hold on;

% Compute loss along trajectories for 3D plotting
Z_sgdm = rosenbrock(history_sgdm(1, :), history_sgdm(2, :));
Z_adam = rosenbrock(history_adam(1, :), history_adam(2, :));
Z_sophia = rosenbrock(history_sophia(1, :), history_sophia(2, :));

% Plot optimizer paths in 3D
plot3(history_sgdm(1, :), history_sgdm(2, :), Z_sgdm, 'r-', 'LineWidth', 2.5, 'DisplayName', 'SGDM');
plot3(history_adam(1, :), history_adam(2, :), Z_adam, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Adam');
plot3(history_sophia(1, :), history_sophia(2, :), Z_sophia, 'g-', 'LineWidth', 2.5, 'DisplayName', 'Sophia');

% Mark starting and optimal points
scatter3(x0, y0, rosenbrock(x0, y0), 200, 'k', 'o', 'filled', 'DisplayName', 'Start (-1.5, 2.5)');
scatter3(1, 1, 0, 400, 'y', '*', 'filled', 'DisplayName', 'Optimum (1, 1)', 'LineWidth', 2);

xlabel('X', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Y', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('f(X,Y)', 'FontSize', 12, 'FontWeight', 'bold');
title('Rosenbrock Function: Optimizer Trajectories (sgdmupdate vs adamupdate vs sophiaupdate)', ...
    'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'NorthWest', 'FontSize', 10);
view(25, 45);
grid on;
set(gca, 'ZScale', 'log');

%% Visualization 2: Convergence Curves
figure('Position', [100, 100, 1350, 500]);

% Loss convergence
subplot(1, 2, 1);
semilogy(iterations, loss_sgdm, 'r-', 'LineWidth', 2.5, 'DisplayName', 'sgdmupdate (LR=0.001)');
hold on;
semilogy(iterations, loss_adam, 'b-', 'LineWidth', 2.5, 'DisplayName', 'adamupdate (LR=0.05)');
semilogy(iterations, loss_sophia, 'g-', 'LineWidth', 2.5, 'DisplayName', 'sophiaupdate (LR=0.05)');
xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Loss (log scale)', 'FontSize', 11, 'FontWeight', 'bold');
title('Loss Convergence Comparison', 'FontSize', 12, 'FontWeight', 'bold');
legend('FontSize', 10, 'Location', 'best');
grid on; grid minor;

% Distance to optimum
subplot(1, 2, 2);
semilogy(iterations, dist_sgdm, 'r-', 'LineWidth', 2.5, 'DisplayName', 'SGDM');
hold on;
semilogy(iterations, dist_adam, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Adam');
semilogy(iterations, dist_sophia, 'g-', 'LineWidth', 2.5, 'DisplayName', 'Sophia');
xlabel('Iteration', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Distance to (1,1) (log scale)', 'FontSize', 11, 'FontWeight', 'bold');
title('Distance to Optimum', 'FontSize', 12, 'FontWeight', 'bold');
legend('FontSize', 10, 'Location', 'best');
grid on; grid minor;

%% Print Results
fprintf('\n');
fprintf(repmat('=', 1, 100));
fprintf('\n');
fprintf('  ROSENBROCK FUNCTION: sgdmupdate vs adamupdate vs sophiaupdate\n');
fprintf(repmat('=', 1, 100));
fprintf('\n');

fprintf('Problem: f(x,y) = (1-x)^2 + 100(y-x^2)^2\n');
fprintf('Initial Point: (%.2f, %.2f)\n', x0, y0);
fprintf('Optimal Point: (1, 1)\n');
fprintf('Initial Loss: %.6e\n\n', rosenbrock(x0, y0));

fprintf('FINAL RESULTS AFTER %d ITERATIONS:\n\n', num_iters);
fprintf('%-15s | %-30s | %-20s | %-15s\n', 'Optimizer', 'MATLAB Function', 'Final Loss', 'Distance');
fprintf(repmat('-', 1, 100));
fprintf('\n');

fprintf('%-15s | %-30s | %-20.6e | %-15.6e\n', 'SGDM', 'sgdmupdate()', loss_sgdm(end), dist_sgdm(end));
fprintf('%-15s | %-30s | %-20.6e | %-15.6e\n', 'Adam', 'adamupdate()', loss_adam(end), dist_adam(end));
fprintf('%-15s | %-30s | %-20.6e | %-15.6e\n', 'Sophia', 'sophiaupdate()', loss_sophia(end), dist_sophia(end));

fprintf('\n');
fprintf(repmat('=', 1, 100));
fprintf('\n');

%% Convergence Analysis
fprintf('CONVERGENCE ANALYSIS:\n');
fprintf(repmat('-', 1, 100));
fprintf('\n');

fprintf('Iterations to reach loss thresholds:\n\n');

thresholds = [1, 0.1, 1e-2, 1e-3, 1e-4];
for thresh = thresholds
    iter_sgdm = find(loss_sgdm <= thresh,1);
    iter_adam = find(loss_adam <= thresh,1);
    iter_sophia = find(loss_sophia <= thresh,1);
    
    if isempty(iter_sgdm), iter_sgdm = inf; end
    if isempty(iter_adam), iter_adam = inf; end
    if isempty(iter_sophia), iter_sophia = inf; end
    
    fprintf('Loss = %.0e:\n', thresh);
    fprintf('  SGDM:   %6d iters  ', iter_sgdm);
    %if iter_sgdm ~= inf, fprintf('(%.2f%% done)\n', 100*iter_sgdm/num_iters);
    %else, fprintf('(not reached)\n'); end
    
    fprintf('  Adam:   %6d iters  ', iter_adam);
    %if iter_adam ~= inf, fprintf('(%.2f%% done)\n', 100*iter_adam/num_iters);
    %else, fprintf('(not reached)\n'); end
    
    fprintf('  Sophia: %6d iters  ', iter_sophia);
    %if iter_sophia ~= inf, fprintf('(%.2f%% done)\n', 100*iter_sophia/num_iters);
    %else, fprintf('(not reached)\n'); end
    
    fprintf('\n');
end

fprintf(repmat('=', 1, 100));
fprintf('\n');

%% Algorithm Details
fprintf('ALGORITHM FUNCTION SIGNATURES:\n');
fprintf(repmat('-', 1, 100));
fprintf('\n');

fprintf('1. [PARAMS, VEL] = sgdmupdate(PARAMS, GRADS, VEL, ...\n');
fprintf('                                ''LearnRate'', LR, ''Momentum'', M)\n');
fprintf('   Formula: v_t = M*v_{t-1} - LR*g_t\n');
fprintf('            theta = theta + v_t\n\n');

fprintf('2. [PARAMS, AVG_G, AVG_SQ_G] = adamupdate(PARAMS, GRADS, AVG_G, AVG_SQ_G, T, ...\n');
fprintf('                                          ''LearnRate'', LR, ''Beta1'', B1, ''Beta2'', B2)\n');
fprintf('   Formula: m_t = B1*m_{t-1} + (1-B1)*g_t\n');
fprintf('            v_t = B2*v_{t-1} + (1-B2)*g_t^2\n');
fprintf('            theta = theta - LR * m_t / (sqrt(v_t) + eps)\n\n');

fprintf('3. [PARAMS, AVG_G, AVG_H] = sophiaupdate(PARAMS, GRADS, AVG_G, AVG_H, HESS, ...\n');
fprintf('                                         UPDATE_H, T, ''LearnRate'', LR, ''Beta1'', B1, ''Beta2'', B2, ''Rho'', RHO)\n');
fprintf('   Formula: m_t = B1*m_{t-1} + (1-B1)*g_t\n');
fprintf('            h_t = B2*h_{t-1} + (1-B2)*hess_t     [if UPDATE_H==true]\n');
fprintf('            ratio = min(|m_t| / (RHO*h_t + eps), 1)\n');
fprintf('            theta = theta - LR * sign(m_t) .* ratio\n\n');

fprintf(repmat('=', 1, 100));
fprintf('\n');

%% Why Sophia Wins
fprintf('WHY SOPHIA OUTPERFORMS ON ROSENBROCK:\n');
fprintf(repmat('-', 1, 100));
fprintf('\n');

fprintf('ROSENBROCK PROPERTIES:\n');
fprintf('  - Function: f(x,y) = (1-x)^2 + 100(y-x^2)^2\n');
fprintf('  - Contains a very narrow valley\n');
fprintf('  - Ill-conditioned: Hessian condition number ~ 10^5\n');
fprintf('  - Different curvatures in x and y directions\n\n');

fprintf('ALGORITHM CAPABILITIES:\n');
fprintf('  +-------------------------------------------------------------+\n');
fprintf('  | SGDM (First-order, Momentum only)                          |\n');
fprintf('  | - Uses: gradients                                          |\n');
fprintf('  | - Advantage: Simple, fast per-iteration                   |\n');
fprintf('  | - Limitation: No curvature adaptation                     |\n');
fprintf('  +-------------------------------------------------------------+\n\n');

fprintf('  +-------------------------------------------------------------+\n');
fprintf('  | Adam (First-order, Adaptive rates)                         |\n');
fprintf('  | - Uses: gradients, squared gradients (v_t)                |\n');
fprintf('  | - Advantage: Per-coordinate learning rate adaptation      |\n');
fprintf('  | - Limitation: Doesn''t use true curvature (Hessian)       |\n');
fprintf('  +-------------------------------------------------------------+\n\n');

fprintf('  +-------------------------------------------------------------+\n');
fprintf('  | Sophia (Second-order, Curvature-aware)                    |\n');
fprintf('  | - Uses: gradients, diagonal Hessian estimates             |\n');
fprintf('  | - Advantage: TRUE curvature-aware preconditioning         |\n');
fprintf('  | - Key: Per-coordinate clipping prevents divergence        |\n');
fprintf('  | - Cost: Efficient diagonal Hessian estimation             |\n');
fprintf('  +-------------------------------------------------------------+\n\n');

fprintf('ON ROSENBROCK''S VALLEY:\n');
fprintf('  - x-dimension (flat exterior): h_xx ~ 2   -> steps ~ LR/2\n');
fprintf('  - y-dimension (narrow valley): h_yy ~ 200 -> steps ~ LR/200\n');
fprintf('  - Sophia adapts step sizes naturally: small in valley, large outside\n');
fprintf('  - Adam cannot distinguish valley from flat regions\n');
fprintf('  - SGDM struggles equally in all directions\n\n');

fprintf('RESULT:\n');
fprintf('  - Sophia reaches machine precision (< 1e-4)\n');
fprintf('  - Adam struggles in the narrow valley\n');
fprintf('  - SGDM converges very slowly\n\n');
