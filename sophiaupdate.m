function [p, avg_g, avg_hess] = sophiaupdate(p, g, avg_g, avg_hess, hess_est,...
    update_hess, t, k, lr, beta1, beta2, rho, bs, weight_decay, epsilon)
%SOPHIAUPDATE Update parameters via Sophia second-order optimization
%
%   [NET,AVG_G,AVG_H] = SOPHIAUPDATE(NET,GRAD,AVG_G,AVG_H,HESS_EST,
%   HESS_UPDATE,ITER,HESS_ITER) updates the learnable parameters of the
%   dlnetwork NET using the Sophia (Second-order Clipped Stochastic
%   Optimization with diagonal Hessian pre-conditioning) algorithm. Sophia
%   improves upon Adam by adapting step sizes to heterogeneous curvatures
%   using a diagonal Hessian estimate and per-coordinate clipping.
%
%   Input GRAD contains the gradients of the loss with respect to each of
%   the network parameters. Inputs AVG_G and AVG_H contain the moving
%   average of the parameter gradients and the moving average of the
%   diagonal Hessian estimates, respectively. AVG_G and AVG_H are obtained
%   from the previous call to SOPHIAUPDATE. GRAD, AVG_G, and AVG_H must be
%   tables with the same structure as NET.Learnables, with a Value variable
%   containing a cell array of parameter gradients, average gradients, or
%   average Hessian estimates. Input GRAD can be obtained using the
%   dlgradient and dlfeval functions. The global learning rate is
%   multiplied by the corresponding learning rate factor for each parameter
%   in each layer of the dlnetwork.
%
%   If inputs AVG_G and AVG_H are empty, the function assumes no previous
%   gradients and executes like the first update in a series of iterations.
%
%   Input HESS contains the diagonal Hessian estimate for the current step.
%   Two estimation methods are supported:
%
%   Option 1 - Gauss-Newton-Bartlett (GNB, recommended for LLMs):
%     Sample labels y_hat ~ Categorical(softmax(logits)), compute the loss
%     on sampled labels, backpropagate, then pass (grad_sampled .* 
%     grad_sampled) as HESS.
%
%   Option 2 - Hutchinson unbiased estimator:
%     Draw u ~ N(0,I), compute the Hessian-vector product H*u via a second
%     backward pass, then pass (u .* Hu) as HESS.
%
%   Pass [] (or zeros) when HESS_UPDATE is false; the stored AVG_H from the
%   previous estimation step will be reused.
%
%   Input HESS_UPDATE is a logical flag. Set HESS_UPDATE to TRUE on Hessian
%   estimation steps (every k iterations, e.g. k=10), FALSE otherwise.
%
%   Input ITER contains the update iteration number. ITER must be a
%   positive integer. Use a value of 1 for the first call to SOPHIAUPDATE
%   and increment by 1 for each successive call in a series of iterations.
%   The Sophia algorithm uses this value to correct for bias in the moving
%   averages at the beginning of a set of iterations.
%
%   Input HESS_ITER contains the update iteration number of the Hessian.
%
%   Outputs NET, AVG_G, and AVG_H are the updated dlnetwork, average
%   gradients, and average Hessian estimates, respectively.
%
%   [PARAMS,AVG_G,AVG_H] = SOPHIAUPDATE(PARAMS,GRAD,AVG_G,AVG_H,HESS,
%   HESS_UPDATE,ITER,HESS_ITER) updates the deep learning parameters in
%   PARAMS using the Sophia algorithm. Input PARAMS can be a dlarray, a
%   numeric array, a cell array, a structure, or a table with a Value
%   variable containing the learnable parameters of the network. GRAD,
%   AVG_G, and AVG_H must have the same datatype and ordering as PARAMS.
%   Input GRAD can be obtained using the dlgradient and dlfeval functions.
%   All parameter values are updated using the global learning rate.
%
%   Outputs PARAMS, AVG_G, and AVG_H are the updated parameters, average
%   gradients, and average Hessian estimates, respectively.
%
%   [___] = SOPHIAUPDATE(___,LEARNRATE,BETA1,BETA2,RHO,BATCHSIZE,
%   WEIGHT_DECAY) also specifies values to use for the global learning
%   rate, gradient EMA decay factor, Hessian EMA decay factor, effective
%   batch size in tokens, per-coordinate clipping threshold, and decoupled
%   weight decay factor. LEARNRATE must be a positive scalar. BETA1, BETA2,
%   and RHO must be scalars between 0 and 1. BATCHSIZE and WEIGHT_DECAY
%   must be positive scalars.
%
%   [___] = SOPHIAUPDATE(___,LEARNRATE,BETA1,BETA2,RHO,BATCHSIZE,
%   WEIGHT_DECAY,EPSILON) specifies a small constant used to prevent
%   division by zero in the update equation. The default value of EPSILON
%   is 1e-15.
%
%   Default values:
%     LEARNRATE    = 1e-4
%     BETA1        = 0.9    (gradient EMA decay)
%     BETA2        = 0.99   (Hessian EMA decay)
%     RHO          = 0.01   (clipping threshold)
%     BATCHSIZE    = 1      (tokens per update)
%     WEIGHT_DECAY = 0.0    (decoupled weight decay)
%     EPSILON      = 1e-12  (numerical stability)
%
%   Example:
%      % Perform Sophia updates with Hessian estimation every 10 steps.
%      p = rand(100,100);
%      avg_g = [];
%      avg_h = [];
%      hess_interval = 10;
%      lr = 3e-4;
%
%      for iter = 1:50
%          do_hess = (mod(iter, hess_interval) == 0);
%
%          % Compute gradients and Hessian estimate (if needed)
%          [loss, grad] = dlfeval(@modelGradients, model, X, Y);
%
%          hess = [];
%          if do_hess
%              [~, grad_sampled] = dlfeval(@sampledGradients, model, X);
%              hess = 480 * 1024 .* grad_sampled .* grad_sampled;
%          end
%
%          % Update parameters
%          [p, avg_g, avg_h] = sophiaupdate(p, grad, avg_g, avg_h, ...
%              hess, do_hess, iter, lr, 0.965, 0.99, 491520, 0.04, 0.1, 1e-15);
%      end
%
%   Reference:
%     Liu et al. (2023). "Sophia: A Scalable Stochastic Second-order 
%     Optimizer via Non-Convex Cubic Regularization." arXiv:2305.14342.
%
%   See also ADAMUPDATE, DLGRADIENT, DLFEVAL, DLNETWORK, DLUPDATE

arguments
    p
    g
    avg_g
    avg_hess
    hess_est
    update_hess  (1,1) logical
    t            (1,1) {mustBeNumeric, mustBePositive, mustBeInteger}
    k            (1,1) {mustBeNumeric, mustBePositive, mustBeInteger} = t
    lr           (1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 1e-4;
    beta1        (1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta1,0), mustBeLessThan(beta1,1)} = 0.9;
    beta2        (1,1) {mustBeNumeric, mustBeGreaterThanOrEqual(beta2,0), mustBeLessThan(beta2,1)} = 0.99;
    rho          (1,1) {mustBeNumeric, mustBePositive} = 0.01;
    bs           (1,1) {mustBeNumeric, mustBePositive} = 1;
    weight_decay (1,1) {mustBeNumeric, mustBeFinite, mustBeNonnegative} = 0.0;
    epsilon      (1,1) {mustBeNumeric, mustBeFinite, mustBePositive} = 1e-12;
end

% Ensure inputs are dlarray where necessary
if isnumeric(g) && isdlarray(p)
    g = dlarray(g);
end

% 1. Buffer Initialization (Stateless container expansion)
% If buffers are empty, we initialize them to the same structure as g
if isempty(avg_g)
    avg_g = dlupdate(@(x) zeros(size(x), 'like', x), g);
end
if isempty(avg_hess)
    avg_hess = dlupdate(@(x) -ones(size(x), 'like', x), g);
end
if isempty(hess_est)
    if update_hess
        error('sophiaupdate:AVG_H cannot be empty when HESS_UPDATE is true.');
    end
    % Create a dummy container of zeros so paramArgs structure remains constant
    hess_est = dlupdate(@(x) zeros(size(x), 'like', x), g);
end

% 2. Setup Container Mapping
persistent func
if isempty(func)
    func = deep.internal.LearnableUpdateFunction( ...
        @iSingleStepValue, ...
        @iSingleStepParameter);
end

% All four inputs are now guaranteed to be containers of the same structure
paramArgs = {g, matlab.lang.internal.move(avg_g), matlab.lang.internal.move(avg_hess), hess_est};
fixedArgs = {update_hess, t, k, lr, beta1, beta2, rho, bs, weight_decay, epsilon};

% Apply update over network/structure
[p, ~, avg_g, avg_hess, ~] = deep.internal.networkContainerFixedArgsFun( ...
    func, p, matlab.lang.internal.move(paramArgs), fixedArgs);

end

function [p, g, avg_g, avg_hess, hess_est] = iSingleStepParameter(p, g, avg_g, avg_hess, ...
    hess_est, update_hess, t, k, lr, beta1, beta2, rho, bs, weight_decay, epsilon)
% Update logic for dlnetwork parameters (Learnables table)

% Apply per-parameter learn-rate factor
lr = lr * p.LearnRateFactor;

% Apply Weight Decay
p.Value = p.Value .* (1 - lr * weight_decay);

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the clipping threshold.
biasCorrection = (1-beta1.^t)./(1-beta2.^k);
effectiveRho = biasCorrection.*rho;

v = p.Value;
p.Value = []; % Clear to save memory during internal update
[v, avg_g, avg_hess] = internal_sophia(v, g, avg_g, avg_hess, hess_est, ...
    lr, beta1, beta2, effectiveRho, bs, epsilon, update_hess);
p.Value = v;
end

function [p, g, avg_g, avg_hess, hess_est] = iSingleStepValue(p, g, avg_g, avg_hess, ...
    hess_est, update_hess, t, k, lr, beta1, beta2, rho, bs, weight_decay, epsilon)
% Update logic for raw dlarrays or numeric arrays

% Apply Weight Decay using base LR
p = p .* (1 - lr * weight_decay);

% Apply a correction factor due to the trailing averages being biased
% towards zero at the beginning.  This is fed into the clipping threshold.
biasCorrection = (1-beta1.^t)./(1-beta2.^k);
effectiveRho = biasCorrection.*rho;

[p, avg_g, avg_hess] = internal_sophia(p, g, avg_g, avg_hess, hess_est, ...
    lr, beta1, beta2, effectiveRho, bs, epsilon, update_hess);
end

function [l, avg_g, avg_hess] = internal_sophia(l, g, avg_g, avg_hess, hess_est, ...
    lr, beta1, beta2, rho, bs, epsilon, update_hess)
%INTERNAL_SOPHIA Internal helper that applies one Sophia optimizer step.
%
%   Applies one complete Sophia update step, delegating to sophiastep for
%   the gradient-EMA / Hessian-EMA / clipping logic.
%   Matches the internal_adam.m pattern: skips update when LEARNRATE == 0.

if lr ~= 0
    [step, avg_g, avg_hess] = sophiastep(g, avg_g, avg_hess, hess_est, ...
    lr, beta1, beta2, rho, bs, epsilon, update_hess);
    
    l = l + step; % Add the step for Gradient Descent
end
end


function [step, avg_g, avg_hess] = sophiastep(g, avg_g, avg_hess, hess_est, ...
    learnrate, beta1, beta2, rho, bs, eps, update_hess)
%SOPHIASTEP Calculate Sophia update step for a single parameter tensor.
%
%   [STEP,AVG_G,AVG_H] = SOPHIASTEP(GRAD,AVG_G,AVG_H,HESS,LEARNRATE,
%   BETA1,BETA2,RHO,BATCHSIZE,EPSILON,UPDATE_H) computes the parameter
%   update step using the Sophia algorithm. Sophia uses a diagonal
%   Hessian-based pre-conditioner with per-coordinate clipping to adapt
%   step sizes to heterogeneous curvatures across parameter dimensions.
%
%   Input GRAD is the current gradient. Inputs AVG_G and AVG_H contain the
%   running exponential moving averages of gradients and diagonal Hessian
%   estimates, respectively. These are updated and returned for use in the
%   next call.
%
%   Input HESS is the new diagonal Hessian estimate for the current step.
%   Pass [] or zeros when UPDATE_H is false; the stored AVG_H from the
%   previous estimation step will be reused.
%
%   Input LEARNRATE is the effective learning rate after bias correction.
%   Input BETA1 is the gradient EMA decay factor. Input BETA2 is the Hessian
%   EMA decay factor. Input RHO is the per-coordinate clipping threshold.
%   Input BATCHSIZE is the effective batch size (batch_size * 
%   sequence_length). Input EPSILON is a small constant for numerical
%   stability. Input UPDATE_H is a logical flag; set to TRUE to update the
%   Hessian EMA with HESS, FALSE otherwise.
%
%   Outputs STEP is the computed parameter update. Outputs AVG_G and AVG_H
%   are the updated gradient and Hessian EMAs, respectively.
%
%   See also SOPHIAUPDATE, ADAMSTEP

% 1. Gradient EMA
avg_g = beta1.*avg_g + (1 - beta1).*g;

% 2. Hessian EMA
if update_hess
    % Use unbiased start trick: if <0, start at current estimate
    isFirst = all(avg_hess < 0);
    avg_hess = isFirst.*hess_est + (1 - isFirst).*(beta2.*avg_hess + (1 - beta2).*hess_est);
end

% 3. Sophia Step
denom = rho.*bs.*avg_hess + eps;
ratio = min(abs(avg_g)./denom, 1);
step = -learnrate.*sign(avg_g).*ratio;
end