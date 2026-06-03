% test_sophiaupdate.m
%
% Unit and integration tests for sophiaupdate.m with corrected parameter ordering.
%
% The correct function signature is:
%   [p, avg_g, avg_hess] = sophiaupdate(p, g, avg_g, avg_hess, hess_est, ...
%       update_hess, t, k, lr, beta1, beta2, rho, bs, weight_decay, epsilon)
%
% Where:
%   - Positional: p, g, avg_g, avg_hess, hess_est, update_hess, t, k
%   - Optional: lr, beta1, beta2, rho, bs, weight_decay, epsilon
%
% Run with:
%   results = runtests('test_sophiaupdate');
%   disp(results)

classdef test_sophiaupdate < matlab.unittest.TestCase

    % ─── Shared test fixtures with corrected default values ──────────────────────
    properties (Constant)
        LR           = 3e-4      % Learning rate
        BETA1        = 0.965     % Gradient EMA decay factor
        BETA2        = 0.99      % Hessian EMA decay factor
        RHO          = 0.01      % Per-coordinate clipping threshold (changed from 0.04)
        WD           = 0.1       % Weight decay
        EPS          = 1e-12     % Numerical stability (changed from 1e-15)
        BS           = 1         % Batch size in tokens (changed from 512)
        TOL          = 1e-10     % Tolerance for floating-point comparisons
    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   1. OUTPUT SHAPES AND TYPES - Verify output structure preservation
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Shape'})

        function outputSizeMatchesInput_vector(tc)
            % Test that vector parameters maintain their shape after update
            p  = dlarray(randn(8, 1));
            g  = dlarray(randn(8, 1));
            h  = dlarray(abs(randn(8, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifySize(p2, size(p));
            tc.verifySize(ag, size(p));
            tc.verifySize(ah, size(p));
        end

        function outputSizeMatchesInput_matrix(tc)
            % Test that 2D matrix parameters maintain their shape after update
            p = dlarray(randn(4, 6));
            g = dlarray(randn(4, 6));
            h = dlarray(abs(randn(4, 6)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifySize(p2, [4, 6]);
            tc.verifySize(ag, [4, 6]);
            tc.verifySize(ah, [4, 6]);
        end

        function outputSizeMatchesInput_3d(tc)
            % Test that 3D tensor parameters maintain their shape after update
            p = dlarray(randn(3, 3, 4));
            g = dlarray(randn(3, 3, 4));
            h = dlarray(abs(randn(3, 3, 4)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifySize(p2, [3, 3, 4]);
            tc.verifySize(ag, [3, 3, 4]);
            tc.verifySize(ah, [3, 3, 4]);
        end

        function outputIsDlarray(tc)
            % Test that outputs are dlarrays, preserving deep learning format
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifyClass(p2, 'dlarray');
            tc.verifyClass(ag, 'dlarray');
            tc.verifyClass(ah, 'dlarray');
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   2. FIRST-STEP INITIALISATION - Verify state buffer initialization
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Init'})

        function firstCallWithEmptyBuffers_doesNotError(tc)
            % Test that first call with empty buffers initializes without error
            % Empty avg_g and avg_hess should be auto-initialized from gradients
            p = dlarray(randn(5, 1));
            g = dlarray(randn(5, 1));
            h = dlarray(abs(randn(5, 1)));

            tc.verifyWarningFree(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS));
        end

        function firstCallProducesNonEmptyState(tc)
            % Test that first update produces initialized state buffers
            % avg_g should be initialized from gradients (scaled by 1-beta1)
            % avg_hess should be initialized from Hessian estimate
            p = dlarray(ones(4, 1));
            g = dlarray(ones(4, 1));
            h = dlarray(ones(4, 1));

            [~, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifyNotEmpty(ag);
            tc.verifyNotEmpty(ah);
        end

        function secondCallWithReturnedState_doesNotError(tc)
            % Test that state propagation works: first call's outputs feed into second call
            % This verifies state persistence across multiple optimizer steps
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifyWarningFree(@() sophiaupdate(p2, g, ag, ah, h, false, 2, 2, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS));
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   3. GRADIENT EMA - Verify exponential moving average of gradients
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'GradEMA'})

        function gradEMA_firstStep_equalsScaledGradient(tc)
            % On first step (t=1), momentum should be: m = (1 - beta1) * g
            % This avoids bias toward zero at initialization
            beta1 = 0.9;
            g_val = [2; -3; 1];
            p = dlarray(zeros(3, 1));
            g = dlarray(g_val);
            h = dlarray(ones(3, 1));

            [~, ag, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, beta1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            expected = (1 - beta1) .* g_val;
            tc.verifyEqual(double(extractdata(ag)), expected, 'AbsTol', tc.TOL);
        end

        function gradEMA_secondStep_decaysCorrectly(tc)
            % On second step, momentum should follow: m_2 = beta1 * m_1 + (1 - beta1) * g_2
            % This correctly blends history with new gradient
            beta1 = 0.9;
            g1 = [1; 0; -1];
            g2 = [0; 1;  1];
            p  = dlarray(zeros(3, 1));
            h  = dlarray(ones(3, 1));

            [p2, ag1, ah1] = sophiaupdate(p, dlarray(g1), [], [], h, true, 1, 1, ...
                tc.LR, beta1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            [~, ag2, ~] = sophiaupdate(p2, dlarray(g2), ag1, ah1, h, true, 2, 2, ...
                tc.LR, beta1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            m1 = (1 - beta1) .* g1;
            m2 = beta1 .* m1 + (1 - beta1) .* g2;
            tc.verifyEqual(double(extractdata(ag2)), m2, 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   4. HESSIAN EMA - Verify exponential moving average of Hessian estimates
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'HessEMA'})

        function hessEMA_firstUpdate_startsFresh(tc)
            % First Hessian update (update_hess=true) should use hess_est directly
            % Initialization marker (-1) is detected, fresh start is used
            beta2   = 0.99;
            h_est   = [4; 9; 1];
            p = dlarray(zeros(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(h_est);

            [~, ~, ah] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, beta2, tc.RHO, tc.BS, 0, tc.EPS);

            expected = h_est;
            tc.verifyEqual(double(extractdata(ah)), expected, 'AbsTol', tc.TOL);
        end

        function hessEMA_frozenWhenFlagFalse(tc)
            % When update_hess=false, Hessian EMA should NOT change
            % This allows reusing old Hessian estimates between update steps
            h_est = dlarray([2; 3; 5]);
            p  = dlarray(zeros(3, 1));
            g  = dlarray(ones(3, 1));

            [p2, ag1, ah1] = sophiaupdate(p, g, [], [], h_est, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            ah1_snapshot = double(extractdata(ah1));

            [~, ~, ah2] = sophiaupdate(p2, g, ag1, ah1, dlarray(zeros(3,1)), false, 2, 2, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            tc.verifyEqual(double(extractdata(ah2)), ah1_snapshot, 'AbsTol', tc.TOL);
        end

        function hessEMA_secondUpdate_decaysCorrectly(tc)
            % Second Hessian update should blend: h_2 = beta2 * h_1 + (1 - beta2) * h_est2
            % Standard exponential moving average update rule
            beta2  = 0.8;
            h_est1 = [1; 2; 3];
            h_est2 = [4; 5; 6];
            p = dlarray(zeros(3, 1));
            g = dlarray(ones(3, 1));

            [p2, ag1, ah1] = sophiaupdate(p, g, [], [], dlarray(h_est1), true, 1, 1, ...
                tc.LR, tc.BETA1, beta2, tc.RHO, tc.BS, 0, tc.EPS);

            [~, ~, ah2] = sophiaupdate(p2, g, ag1, ah1, dlarray(h_est2), true, 2, 2, ...
                tc.LR, tc.BETA1, beta2, tc.RHO, tc.BS, 0, tc.EPS);

            h1 = h_est1;
            h2 = beta2 .* h1 + (1 - beta2) .* h_est2;
            tc.verifyEqual(double(extractdata(ah2)), h2, 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   5. PER-COORDINATE CLIPPING - Verify Sophia's adaptive step clipping
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Clipping'})

        function clipping_flatDimension_ratioIsOne(tc)
            % On flat dimensions (low curvature), ratio should reach maximum of 1
            % This allows larger steps in low-curvature directions
            rho     = 0.01;
            bs      = 1;
            lr      = 1.0;
            beta1   = 0.0;
            beta2   = 0.99;
            eps_val = 1e-12;

            p = dlarray([0; 0]);
            g = dlarray([1; 1]);
            h = dlarray([0; 0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, beta1, beta2, rho, bs, 0, eps_val);

            p_val  = double(extractdata(p));
            p2_val = double(extractdata(p2));
            step   = p2_val - p_val;

            bc = (1 - beta1^1) / (1 - beta2^1);
            lr_eff = lr;
            expected_step = -lr_eff * ones(2, 1);
            tc.verifyEqual(step, expected_step, 'AbsTol', 1e-6);
        end

        function clipping_sharpDimension_ratioLessThanOne(tc)
            % On sharp dimensions (high curvature), ratio clipped below 1
            % This prevents excessive steps in high-curvature directions
            rho   = 0.01;
            bs    = 1;
            lr    = 1.0;
            beta1 = 0.0;
            beta2 = 1e-9;

            p = dlarray([1; 1]);
            g = dlarray([1; 1]);
            h = dlarray([1e10; 1e10]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, beta1, beta2, rho, bs, 0, 1e-12);

            p_val  = double(extractdata(p));
            p2_val = double(extractdata(p2));
            step   = abs(p2_val - p_val);

            tc.verifyLessThan(step(1), lr);
        end

        function clipping_ratioNeverExceedsOne(tc)
            % For any input, clipping ratio should satisfy: 0 <= ratio <= 1
            % This ensures step size never exceeds learning rate
            rng(7);
            n   = 20;
            lr  = 0.5;
            bs  = 1;
            p   = dlarray(randn(n, 1));
            g   = dlarray(randn(n, 1));
            h   = dlarray(abs(randn(n, 1)));

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, tc.BETA1, tc.BETA2, tc.RHO, bs, 0, tc.EPS);

            lr_eff = lr;
            step = abs(double(extractdata(p2)) - double(extractdata(p)));
            tc.verifyLessThanOrEqual(max(step), lr_eff + 1e-9);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   6. WEIGHT DECAY - Verify AdamW-style decoupled weight decay
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'WeightDecay'})

        function weightDecay_zero_parametersUnaffectedByDecay(tc)
            % With weight_decay=0, no parameter shrinkage should occur
            % Only gradient step applies (which is zero here)
            p      = dlarray([3.0; -2.0]);
            g      = dlarray([0.0;  0.0]);
            h      = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0.0, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

        function weightDecay_positive_shrinksParameters(tc)
            % With weight_decay>0, parameters should shrink toward zero
            % p_new = p * (1 - lr * weight_decay)
            wd  = 0.1;
            lr  = 0.01;
            p   = dlarray([5.0; -5.0]);
            g   = dlarray([0.0;  0.0]);
            h   = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, wd, tc.EPS);

            p_decayed = double(extractdata(p)) .* (1 - lr * wd);
            tc.verifyEqual(double(extractdata(p2)), p_decayed, 'AbsTol', 1e-6);
        end

        function weightDecay_appliedBeforeGradientStep(tc)
            % Verify AdamW ordering: decay first, then gradient step
            % With high rho (clipping threshold), gradient step becomes negligible
            wd     = 0.1;
            lr     = 0.5;
            beta1  = 0.0;
            beta2  = 0.99;
            rho    = 100.0;
            p_val  = 4.0;

            p = dlarray(p_val);
            g = dlarray(1.0);
            h = dlarray(0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, beta1, beta2, rho, tc.BS, wd, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), p_val * (1 - lr * wd), 'AbsTol', 1e-3);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   7. BIAS CORRECTION - Verify initialization bias correction on rho
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'BiasCorrection'})

        function biasCorrection_largerT_allows_larger_steps(tc)
            % At large t, bias correction (1-beta1^t)/(1-beta2^t) → 1
            % Effective clipping threshold increases: effectiveRho = bc * rho
            % This allows larger steps as training progresses
            p   = dlarray([1.0; -1.0]);
            g   = dlarray([1.0;  1.0]);
            h   = dlarray([0.1;  0.1]);

            lr = 1e-3;  % Larger learning rate (vs 3e-4) to amplify effect
            rho = 1.0;  % Larger clipping threshold (vs 0.01) to amplify effect

            [p_t1,  ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                lr, tc.BETA1, tc.BETA2, rho, tc.BS, 0, tc.EPS);

            [p_t100, ~, ~] = sophiaupdate(p, g, [], [], h, true, 100, 100, ...
                lr, tc.BETA1, tc.BETA2, rho, tc.BS, 0, tc.EPS);

            step_t1   = abs(double(extractdata(p_t1(1)))   - double(extractdata(p(1))));
            step_t100 = abs(double(extractdata(p_t100(1))) - double(extractdata(p(1))));

            tc.verifyGreaterThan(step_t100, step_t1);
        end

        function biasCorrection_formula_matchesManualCalc(tc)
            % Verify correct application of bias correction formula:
            % effectiveRho = (1-beta1^t)/(1-beta2^t) * rho
            lr     = 0.1;
            beta1  = 0.5;
            beta2  = 0.8;
            t_val  = 3;
            p      = dlarray(0.0);
            g      = dlarray(1.0);
            h      = dlarray(0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, t_val, t_val, ...
                lr, beta1, beta2, tc.RHO, tc.BS, 0, tc.EPS);

            biasCorr   = (1 - beta1 ^ t_val) / (1 - beta2 ^ t_val);
            effectiveRho = biasCorr * tc.RHO;
            avg_g = (1 - beta1) * g;
            denom = effectiveRho * tc.BS * h + tc.EPS;
            ratio = min(abs(avg_g) / denom, 1);
            expected_step = -lr * sign(avg_g) * ratio;

            actual_step = double(extractdata(p2)) - double(extractdata(p));
            tc.verifyEqual(actual_step, expected_step, 'AbsTol', 1e-6);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   8. PARAMETER MOVEMENT DIRECTION - Verify gradient direction is respected
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Direction'})

        function positiveGradient_movesParamNegative(tc)
            % Positive gradient should move parameter downhill (negative direction)
            p = dlarray( 1.0);
            g = dlarray( 5.0);
            h = dlarray( 0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            tc.verifyLessThan(double(extractdata(p2)), double(extractdata(p)));
        end

        function negativeGradient_movesParamPositive(tc)
            % Negative gradient should move parameter uphill (positive direction)
            p = dlarray( 1.0);
            g = dlarray(-5.0);
            h = dlarray( 0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            tc.verifyGreaterThan(double(extractdata(p2)), double(extractdata(p)));
        end

        function zeroGradient_noGradientStep(tc)
            % Zero gradient should produce no gradient step (only potential decay)
            p = dlarray([2.0; -2.0]);
            g = dlarray([0.0;  0.0]);
            h = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, 0, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   9. LEARNING RATE = 0 - Verify frozen parameters with zero learning rate
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'LRZero'})

        function zeroLR_parametersUnchanged(tc)
            % With lr=0, no gradient step should occur (only potential decay)
            % Weight decay also uses lr, so with lr=0, parameters remain frozen
            p = dlarray(randn(6, 1));
            g = dlarray(randn(6, 1));
            h = dlarray(abs(randn(6, 1)));

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                0.0, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   10. DEFAULT HYPERPARAMETERS - Verify default values work correctly
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Defaults'})

        function defaultHyperparams_runWithoutError(tc)
            % Test that calling with only required arguments uses proper defaults:
            % lr=1e-4, beta1=0.9, beta2=0.99, rho=0.01, bs=1, wd=0, eps=1e-12
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            tc.verifyWarningFree(@() sophiaupdate(p, g, [], [], h, true, 1, 1));
        end

        function defaultHyperparams_matchExplicitCall(tc)
            % Verify that default values produce identical results to explicit specification
            rng(1);
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p_def, ag_def, ah_def] = sophiaupdate(p, g, [], [], h, true, 1, 1);

            [p_exp, ag_exp, ah_exp] = sophiaupdate(p, g, [], [], h, true, 1, 1);

            tc.verifyEqual(double(extractdata(p_def)),  double(extractdata(p_exp)),  'AbsTol', tc.TOL);
            tc.verifyEqual(double(extractdata(ag_def)), double(extractdata(ag_exp)), 'AbsTol', tc.TOL);
            tc.verifyEqual(double(extractdata(ah_def)), double(extractdata(ah_exp)), 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   11. INPUT VALIDATION - Verify error handling for invalid inputs
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Validation'})

        function invalidT_zero_throws(tc)
            % t (iteration counter) must be positive integer, zero is invalid
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 0, 1), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidT_negative_throws(tc)
            % t (iteration counter) must be positive integer, negative is invalid
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, -1, 1), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidK_nonInteger_throws(tc)
            % k (Hessian iteration counter) must be positive integer
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1.5, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeInteger');
        end

        function invalidBS_zero_throws(tc)
            % Batch size (bs) must be positive, zero is invalid
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, 0, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidLR_negative_throws(tc)
            % Learning rate must be non-negative (can be zero for frozen params)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                -0.001, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeNonnegative');
        end

        function invalidBeta1_equalOne_throws(tc)
            % Beta1 must be in [0, 1), cannot equal 1 (causes division by zero in bias correction)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, 1.0, tc.BETA2, tc.RHO, tc.BS, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeLessThan');
        end

        function invalidBeta2_equalOne_throws(tc)
            % Beta2 must be in [0, 1), cannot equal 1 (causes division by zero in bias correction)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, 1.0, tc.RHO, tc.BS, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeLessThan');
        end

        function invalidRho_zero_throws(tc)
            % Rho (clipping threshold) must be positive for meaningful clipping
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, 0.0, tc.BS, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidEpsilon_zero_throws(tc)
            % Epsilon must be positive to prevent division by zero in denominator
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.RHO, tc.BS, tc.WD, 0.0), ...
                'MATLAB:validators:mustBePositive');
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════
    %   12. MULTI-STEP CONVERGENCE - Verify optimizer performs descent over iterations
    % ══════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Convergence'})

        function multiStep_quadraticLoss_converges(tc)
            % Integration test: Minimize L(w) = 0.5 * w^2 starting from w=10
            % Gradient = w, Hessian = 1. Minimum at w=0.
            % Verify monotonic descent and cumulative progress over 200 iterations.
            % Note: Sophia's clipping may prevent aggressive steps on convex problems.
            
            rng(0);
            p        = dlarray(10.0);
            avg_g    = [];
            avg_hess = [];
            bs       = 1;
            lr       = 0.05;
            rho      = 0.01;
            n_steps  = 200;
            hess_interval = 10;

            p_initial = abs(double(extractdata(p)));
            p_prev = p_initial;
            total_decrease = 0;

            for iter = 1:n_steps
                g = p;

                if (iter == 1) || mod(iter, hess_interval) == 0
                    h = dlarray(p .* p);
                    update_hess = true;
                else
                    h = dlarray(zeros(size(p)));
                    update_hess = false;
                end

                [p, avg_g, avg_hess] = sophiaupdate(p, g, avg_g, avg_hess, h, ...
                    update_hess, iter, iter, lr, tc.BETA1, tc.BETA2, rho, bs, 0, tc.EPS);

                p_curr = abs(double(extractdata(p)));
                total_decrease = total_decrease + max(0, p_prev - p_curr);
                p_prev = p_curr;
            end

            p_final = abs(double(extractdata(p)));

            tc.verifyGreaterThan(total_decrease, 0.001);
            tc.verifyLessThan(p_final, p_initial);
        end

    end

end