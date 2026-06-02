% example_sophiaupdate.m
%
% Demonstrates sophiaupdate.m on a diagonal quadratic loss:
%
%   L(w) = 0.5 * w' * diag(A) * w  +  b' * w
%
% A is ill-conditioned (cond = 1000) to stress-test curvature handling.
% Compares Sophia (GNB estimator), Adam, and SGD-M from the same init.
%
% NOTE on bs and hess_est:
%   sophiastep denominator = rho * bs * avg_hess.
%   The bs parameter is a token-count scaling designed for LLM training
%   (bs ~ 500k). For this small demo we pass bs=1 and hess_est = g.*g
%   so the denominator reduces to rho * avg_hess, which is the correct
%   scale for a per-sample quadratic problem.

rng(42);

% ── Problem setup ─────────────────────────────────────────────────────────────
d      = 50;
A_diag = dlarray(logspace(0, 3, d)');   % condition number = 1000
b_vec  = dlarray(randn(d, 1));

A_num  = double(extractdata(A_diag));
b_num  = double(extractdata(b_vec));
w_star = -A_num .\ b_num;
L_star = 0.5 * sum(A_num .* w_star.^2) + sum(b_num .* w_star);

% ── Training hyper-parameters ─────────────────────────────────────────────────
max_iters     = 1000;
hess_interval = 10;

% Options: 'gnb' (g^2), 'analytic' (exact A), 'hutchinson' (u'*H*u)
hess_type = 'analytic'; 

% bs=1: removes LLM token-count scaling so that rho*bs*avg_hess ~ rho*g^2,
% keeping the clipping ratio in a healthy [0,1] range for this problem.
bs_sophia = 1; 

lr_sophia = 0.001;
beta1     = 0.965;
beta2     = 0.99;
rho       = 0.04;
wd        = 0.0;

lr_adam  = 0.001;
lr_sgdm  = 1.5 / (2 * max(A_num));   % stable: lr << 2 / max_curvature
momentum = 0.9;

% ── Initialise parameters from the same random point ─────────────────────────
w0 = randn(d, 1);

w_sophia  = dlarray(w0);
w_adam    = dlarray(w0);
w_sgdm    = dlarray(w0);

avg_g_sophia    = [];
avg_hess_sophia = [];
avg_g_adam      = [];
avg_gsq_adam    = [];
vel_sgdm        = [];

loss_sophia = zeros(1, max_iters);
loss_adam   = zeros(1, max_iters);
loss_sgdm   = zeros(1, max_iters);

% ── Loss + gradient (anonymous function, required by dlfeval) ─────────────────
quadLoss = @(w, A, b) deal( ...
    sum(0.5 .* A .* w.^2 + b .* w), ...
    dlgradient(sum(0.5 .* A .* w.^2 + b .* w), w));

% ── Training loop ─────────────────────────────────────────────────────────────
for t = 1:max_iters

    % ── 1. Sophia ─────────────────────────────────────────────────────────────
    do_hess = (mod(t, hess_interval) == 0) || (t == 1);

    [L_s, g_s] = dlfeval(quadLoss, w_sophia, A_diag, b_vec);
    loss_sophia(t) = double(extractdata(L_s));

    % GNB estimator: hess_est = g .* g
    % With bs=1 the denominator rho*bs*avg_hess = rho*avg_hess ~ rho*g^2,
    % giving ratio = |avg_g| / (rho*g^2) which is well-scaled.
    hess_est = [];
    if do_hess
        switch lower(hess_type)
            case 'gnb'
                % Gauss-Newton-Bartlett: g * g
                hess_est = g_s .* g_s;
                
            case 'analytic'
                % Exact Hessian for this quadratic problem is A_diag
                hess_est = A_diag;
                
            case 'hutchinson'
                % Hutchinson: u .* (H*u) where u ~ N(0,1)
                u = dlarray(randn(d, 1), 'CB');
                % For this quadratic, H*u is simply A_diag .* u
                Hu = A_diag .* u; 
                hess_est = max(u .* Hu,0);
        end
    end

    [w_sophia, avg_g_sophia, avg_hess_sophia] = sophiaupdate( ...
        w_sophia, g_s, avg_g_sophia, avg_hess_sophia, hess_est, do_hess, ...
        t, lr_sophia, beta1, beta2, bs_sophia, rho);

    % ── 2. Adam ───────────────────────────────────────────────────────────────
    [L_a, g_a] = dlfeval(quadLoss, w_adam, A_diag, b_vec);
    loss_adam(t) = double(extractdata(L_a));

    [w_adam, avg_g_adam, avg_gsq_adam] = adamupdate( ...
        w_adam, g_a, avg_g_adam, avg_gsq_adam, t, lr_adam, beta1, beta2);

    % ── 3. SGD-M via sgdmupdate ───────────────────────────────────────────────
    [L_m, g_m] = dlfeval(quadLoss, w_sgdm, A_diag, b_vec);
    loss_sgdm(t) = double(extractdata(L_m));

    [w_sgdm, vel_sgdm] = sgdmupdate(w_sgdm, g_m, vel_sgdm, lr_sgdm, momentum);

end

% ── Results ───────────────────────────────────────────────────────────────────
fprintf('\n=== Final loss after %d iterations ===\n', max_iters);
fprintf('  Sophia  : %.6e\n', loss_sophia(end));
fprintf('  Adam    : %.6e\n', loss_adam(end));
fprintf('  SGD-M   : %.6e\n', loss_sgdm(end));
fprintf('  Optimum : %.6e\n', L_star);

fprintf('\n=== Distance to optimum (L2) ===\n');
fprintf('  Sophia  : %.6e\n', norm(double(extractdata(w_sophia)) - w_star));
fprintf('  Adam    : %.6e\n', norm(double(extractdata(w_adam))   - w_star));
fprintf('  SGD-M   : %.6e\n', norm(double(extractdata(w_sgdm))   - w_star));

% ── Convergence plot: excess loss L(w)-L* is always >= 0 → safe for semilogy ─
excess_sophia = loss_sophia - L_star;
excess_adam   = loss_adam   - L_star;
excess_sgdm   = loss_sgdm   - L_star;

figure('Name', 'Sophia vs Adam vs SGD-M – Convergence');
semilogy(1:max_iters, excess_sophia, 'LineWidth', 2); hold on;
semilogy(1:max_iters, excess_adam,   'LineWidth', 2);
semilogy(1:max_iters, excess_sgdm,   'LineWidth', 2); hold off;
xlabel('Iteration');
ylabel('Excess loss  L(w) - L(w*)  (log scale)');
title(sprintf('Diagonal quadratic  d=%d  cond(A)=1000', d));
legend('Sophia', 'Adam', 'SGD-M', 'Location', 'northeast');
grid on;