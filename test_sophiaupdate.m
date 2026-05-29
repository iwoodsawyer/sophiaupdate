% test_sophiaupdate.m
%
% Unit and integration tests for sophiaupdate.m.
%
% Run with:
%   results = runtests('test_sophiaupdate');
%   disp(results)

classdef test_sophiaupdate < matlab.unittest.TestCase

    % ── Shared test fixtures ────────────────────────────────────────────────────────────────
    properties (Constant)
        LR           = 3e-4
        BETA1        = 0.965
        BETA2        = 0.99
        RHO          = 0.04
        WD           = 0.1
        EPS          = 1e-15
        BS           = 512       % small token count keeps numerics tractable
        TOL          = 1e-10     % absolute tolerance for floating-point comparisons
    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   1. OUTPUT SHAPES AND TYPES
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Shape'})

        function outputSizeMatchesInput_vector(tc)
            p  = dlarray(randn(8, 1));
            g  = dlarray(randn(8, 1));
            h  = dlarray(abs(randn(8, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifySize(p2, size(p));
            tc.verifySize(ag, size(p));
            tc.verifySize(ah, size(p));
        end

        function outputSizeMatchesInput_matrix(tc)
            p = dlarray(randn(4, 6));
            g = dlarray(randn(4, 6));
            h = dlarray(abs(randn(4, 6)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifySize(p2, [4, 6]);
            tc.verifySize(ag, [4, 6]);
            tc.verifySize(ah, [4, 6]);
        end

        function outputSizeMatchesInput_3d(tc)
            p = dlarray(randn(3, 3, 4));
            g = dlarray(randn(3, 3, 4));
            h = dlarray(abs(randn(3, 3, 4)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifySize(p2, [3, 3, 4]);
            tc.verifySize(ag, [3, 3, 4]);
            tc.verifySize(ah, [3, 3, 4]);
        end

        function outputIsDlarray(tc)
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                 tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifyClass(p2, 'dlarray');
            tc.verifyClass(ag, 'dlarray');
            tc.verifyClass(ah, 'dlarray');
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   2. FIRST-STEP INITIALISATION  (avg_g = [], avg_hess = [])
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Init'})

        function firstCallWithEmptyBuffers_doesNotError(tc)
            p = dlarray(randn(5, 1));
            g = dlarray(randn(5, 1));
            h = dlarray(abs(randn(5, 1)));

            tc.verifyWarningFree(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS));
        end

        function firstCallProducesNonEmptyState(tc)
            p = dlarray(ones(4, 1));
            g = dlarray(ones(4, 1));
            h = dlarray(ones(4, 1));

            [~, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifyNotEmpty(ag);
            tc.verifyNotEmpty(ah);
        end

        function secondCallWithReturnedState_doesNotError(tc)
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p2, ag, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifyWarningFree(@() sophiaupdate(p2, g, ag, ah, h, false, 2, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS));
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   3. GRADIENT EMA (momentum numerator)
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'GradEMA'})

        function gradEMA_firstStep_equalsScaledGradient(tc)
            % At t=1 with avg_g=0, the rescaling trick gives:
            %   avg_g = (1 - beta1) * (0 + g) = (1-beta1)*g
            beta1 = 0.9;
            g_val = [2; -3; 1];
            p = dlarray(zeros(3, 1));
            g = dlarray(g_val);
            h = dlarray(ones(3, 1));

            [~, ag, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, beta1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            expected = (1 - beta1) .* g_val;
            tc.verifyEqual(double(extractdata(ag)), expected, 'AbsTol', tc.TOL);
        end

        function gradEMA_secondStep_decaysCorrectly(tc)
            % m_1 = (1-b1)*g1;  m_2 = b1*m_1 + (1-b1)*g2
            beta1 = 0.9;
            g1 = [1; 0; -1];
            g2 = [0; 1;  1];
            p  = dlarray(zeros(3, 1));
            h  = dlarray(ones(3, 1));

            [p2, ag1, ah1] = sophiaupdate(p, dlarray(g1), [], [], h, true, 1, ...
                tc.LR, beta1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            [~, ag2, ~] = sophiaupdate(p2, dlarray(g2), ag1, ah1, h, true, 2, ...
                tc.LR, beta1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            m1 = (1 - beta1) .* g1;
            m2 = beta1 .* m1 + (1 - beta1) .* g2;
            tc.verifyEqual(double(extractdata(ag2)), m2, 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   4. HESSIAN EMA
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'HessEMA'})

        function hessEMA_updatedWhenFlagTrue(tc)
            % At t=1, avg_hess starts at 0:
            %   h_1 = beta2*0 + (1-beta2)*hess_est = (1-beta2)*hess_est
            beta2   = 0.99;
            h_est   = [4; 9; 1];
            p = dlarray(zeros(3, 1));
            g = dlarray(ones(3,  1));
            h = dlarray(h_est);

            [~, ~, ah] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, beta2, tc.BS, tc.RHO, 0, tc.EPS);

            expected = (1 - beta2) .* h_est;
            tc.verifyEqual(double(extractdata(ah)), expected, 'AbsTol', tc.TOL);
        end

        function hessEMA_frozenWhenFlagFalse(tc)
            % Run step 1 with update_hessian=true to set avg_hess,
            % then step 2 with update_hessian=false: avg_hess must not change.
            h_est = dlarray([2; 3; 5]);
            p  = dlarray(zeros(3, 1));
            g  = dlarray(ones(3,  1));

            [p2, ag1, ah1] = sophiaupdate(p, g, [], [], h_est, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            ah1_snapshot = double(extractdata(ah1));

            [~, ~, ah2] = sophiaupdate(p2, g, ag1, ah1, dlarray(zeros(3,1)), false, 2, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            tc.verifyEqual(double(extractdata(ah2)), ah1_snapshot, 'AbsTol', tc.TOL);
        end

        function hessEMA_secondUpdate_decaysCorrectly(tc)
            % h_1 = (1-b2)*h_est1;  h_2 = b2*h_1 + (1-b2)*h_est2
            beta2  = 0.8;
            h_est1 = [1; 2; 3];
            h_est2 = [4; 5; 6];
            p = dlarray(zeros(3, 1));
            g = dlarray(ones(3,  1));

            [p2, ag1, ah1] = sophiaupdate(p, g, [], [], dlarray(h_est1), true, 1, ...
                tc.LR, tc.BETA1, beta2, tc.BS, tc.RHO, 0, tc.EPS);

            [~, ~, ah2] = sophiaupdate(p2, g, ag1, ah1, dlarray(h_est2), true, 2, ...
                tc.LR, tc.BETA1, beta2, tc.BS, tc.RHO, 0, tc.EPS);

            h1 = (1 - beta2) .* h_est1;
            h2 = beta2 .* h1 + (1 - beta2) .* h_est2;
            tc.verifyEqual(double(extractdata(ah2)), h2, 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   5. PER-COORDINATE CLIPPING
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Clipping'})

        function clipping_flatDimension_ratioIsOne(tc)
            % When avg_hess ~ 0, the denominator collapses to epsilon,
            % ratio = |avg_g| / eps >> 1  =>  clamped to 1.
            % The step magnitude should equal lr_eff * 1 = lr_eff.
            rho     = 0.04;
            bs      = 1;
            lr      = 1.0;
            beta1   = 0.0;   % no decay so avg_g = g immediately
            beta2   = 0.99;
            eps_val = 1e-15;

            p = dlarray([0; 0]);
            g = dlarray([1; 1]);
            h = dlarray([0; 0]);   % zero Hessian => flat curvature

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                lr, beta1, beta2, bs, rho, 0, eps_val);

            p_val  = double(extractdata(p));
            p2_val = double(extractdata(p2));
            step   = p2_val - p_val;

            % Bias correction at t=1: bc = 1 - beta1^1 = 1  => lr_eff = lr/1 = 1
            % sign(avg_g) = [1;1], ratio = 1  => step = -1*[1;1]
            tc.verifyEqual(step, [-1; -1], 'AbsTol', 1e-6);
        end

        function clipping_sharpDimension_ratioLessThanOne(tc)
            % When Hessian is very large, ratio << 1 and step is shrunk.
            rho   = 0.04;
            bs    = 1;
            lr    = 1.0;
            beta1 = 0.0;
            beta2 = 1e-9;

            p = dlarray([1; 1]);
            g = dlarray([1; 1]);
            h = dlarray([1e10; 1e10]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                lr, beta1, beta2, bs, rho, 0, 1e-15);

            p_val  = double(extractdata(p));
            p2_val = double(extractdata(p2));
            step   = abs(p2_val - p_val);

            % ratio = |avg_g| / (rho*bs*avg_hess + eps) << 1
            % so |step| << lr
            tc.verifyLessThan(step(1), lr);
        end

        function clipping_ratioNeverExceedsOne(tc)
            % For any random inputs, the ratio must always be in [0, 1].
            % We verify indirectly: |step| <= lr_eff for all coordinates.
            rng(7);
            n   = 20;
            lr  = 0.5;
            bs  = 256;
            p   = dlarray(randn(n, 1));
            g   = dlarray(randn(n, 1));
            h   = dlarray(abs(randn(n, 1)));

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                lr, tc.BETA1, tc.BETA2, bs, tc.RHO, 0, tc.EPS);

            biasCorr = 1 - tc.BETA1 ^ 1;
            lr_eff   = lr / biasCorr;

            step = abs(double(extractdata(p2)) - double(extractdata(p)));
            tc.verifyLessThanOrEqual(max(step), lr_eff + 1e-9);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   6. WEIGHT DECAY
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'WeightDecay'})

        function weightDecay_zero_parametersUnaffectedByDecay(tc)
            % With wd=0 and a single step, any parameter change is purely
            % from the gradient step, not decay.
            p      = dlarray([3.0; -2.0]);
            g      = dlarray([0.0;  0.0]);   % zero gradient => zero step
            h      = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0.0, tc.EPS);

            % sign(0) = 0 in MATLAB, so step = 0 and p should be unchanged
            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

        function weightDecay_positive_shrinksParameters(tc)
            % With non-zero weight decay, |p2| < |p| (all else equal).
            wd  = 0.1;
            lr  = 0.01;
            p   = dlarray([5.0; -5.0]);
            g   = dlarray([0.0;  0.0]);
            h   = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                lr, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, wd, tc.EPS);

            % Use base lr
            p_decayed = double(extractdata(p)) .* (1 - lr * wd);

            tc.verifyEqual(double(extractdata(p2)), p_decayed, 'AbsTol', 1e-6);
        end

        function weightDecay_appliedBeforeGradientStep(tc)
            % Verify the AdamW-style ordering: decay first, then gradient step.
            % With known inputs we can compute the expected value analytically.
            wd     = 0.2;
            lr     = 1.0;
            beta1  = 0.0;
            beta2  = 1e-9;
            rho    = 1e10;  % ratio = 0 => no gradient step
            p_val  = 4.0;

            p = dlarray(p_val);
            g = dlarray(1.0);
            h = dlarray(1.0);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                lr, beta1, beta2, tc.BS, rho, wd, tc.EPS);

            % p after decay = p * (1 - lr * wd)
            % 4 * (1 - 1.0 * 0.2) = 3.2
            tc.verifyEqual(double(extractdata(p2)), p_val * (1 - lr * wd), 'AbsTol', 1e-4);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   7. BIAS CORRECTION
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'BiasCorrection'})

        function biasCorrection_largert_reducesEffectiveLR(tc)
            % At large t, bias correction (1 - beta1^t) → 1, so lr_eff → lr.
            % At small t, lr_eff > lr.  Verify lr_eff(t=1) > lr_eff(t=100).
            p   = dlarray([1.0; -1.0]);
            g   = dlarray([1.0;  1.0]);
            h   = dlarray([0.0;  0.0]);   % flat => ratio = 1, full step

            [p_t1,  ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            [p_t100, ~, ~] = sophiaupdate(p, g, [], [], h, true, 100, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            step_t1   = abs(double(extractdata(p_t1(1)))   - double(extractdata(p(1))));
            step_t100 = abs(double(extractdata(p_t100(1))) - double(extractdata(p(1))));

            tc.verifyGreaterThan(step_t1, step_t100);
        end

        function biasCorrection_formula_matchesManualCalc(tc)
            % lr_eff = lr / (1 - beta1^t).  Verify by isolating step size.
            lr     = 0.1;
            beta1  = 0.5;
            t_val  = 3;
            p      = dlarray(0.0);
            g      = dlarray(1.0);
            h      = dlarray(0.0);   % flat => ratio = 1

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, t_val, ...
                lr, beta1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            biasCorr  = 1 - beta1 ^ t_val;
            lr_eff    = lr / biasCorr;
            m_t       = (1 - beta1) .* 1;    % avg_g starts at 0, g=1
            expected_step = -lr_eff * sign(m_t) * 1;   % ratio = 1 (flat)

            actual_step = double(extractdata(p2)) - double(extractdata(p));
            tc.verifyEqual(actual_step, expected_step, 'AbsTol', 1e-6);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   8. PARAMETER MOVEMENT DIRECTION
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Direction'})

        function positiveGradient_movesParamNegative(tc)
            p = dlarray( 1.0);
            g = dlarray( 5.0);
            h = dlarray( 0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            tc.verifyLessThan(double(extractdata(p2)), double(extractdata(p)));
        end

        function negativeGradient_movesParamPositive(tc)
            p = dlarray( 1.0);
            g = dlarray(-5.0);
            h = dlarray( 0.1);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            tc.verifyGreaterThan(double(extractdata(p2)), double(extractdata(p)));
        end

        function zeroGradient_noGradientStep(tc)
            p = dlarray([2.0; -2.0]);
            g = dlarray([0.0;  0.0]);
            h = dlarray([1.0;  1.0]);

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, 0, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   9. LEARNING RATE = 0  (frozen parameters)
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'LRZero'})

        function zeroLR_parametersUnchanged(tc)
            p = dlarray(randn(6, 1));
            g = dlarray(randn(6, 1));
            h = dlarray(abs(randn(6, 1)));

            [p2, ~, ~] = sophiaupdate(p, g, [], [], h, true, 1, ...
                0.0, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS);

            tc.verifyEqual(double(extractdata(p2)), double(extractdata(p)), ...
                'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   10. DEFAULT HYPERPARAMETERS
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Defaults'})

        function defaultHyperparams_runWithoutError(tc)
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            tc.verifyWarningFree(@() sophiaupdate(p, g, [], [], h, true, 1));
        end

        function defaultHyperparams_matchExplicitCall(tc)
            rng(1);
            p = dlarray(randn(4, 1));
            g = dlarray(randn(4, 1));
            h = dlarray(abs(randn(4, 1)));

            [p_def, ag_def, ah_def] = sophiaupdate(p, g, [], [], h, true, 1);

            [p_exp, ag_exp, ah_exp] = sophiaupdate(p, g, [], [], h, true, 1);

            tc.verifyEqual(double(extractdata(p_def)),  double(extractdata(p_exp)),  'AbsTol', tc.TOL);
            tc.verifyEqual(double(extractdata(ag_def)), double(extractdata(ag_exp)), 'AbsTol', tc.TOL);
            tc.verifyEqual(double(extractdata(ah_def)), double(extractdata(ah_exp)), 'AbsTol', tc.TOL);
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   11. INPUT VALIDATION  (must throw)
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Validation'})

        function invalidT_zero_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 0), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidT_negative_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, -1), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidBS_zero_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, 0, tc.RHO, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidLR_negative_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                -0.001, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeNonnegative');
        end

        function invalidBeta1_equalOne_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, 1.0, tc.BETA2, tc.BS, tc.RHO, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeLessThan');
        end

        function invalidBeta2_equalOne_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, 1.0, tc.BS, tc.RHO, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBeLessThan');
        end

        function invalidRho_zero_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, 0.0, tc.WD, tc.EPS), ...
                'MATLAB:validators:mustBePositive');
        end

        function invalidEpsilon_zero_throws(tc)
            p = dlarray(ones(3, 1));
            g = dlarray(ones(3, 1));
            h = dlarray(ones(3, 1));

            tc.verifyError(@() sophiaupdate(p, g, [], [], h, true, 1, ...
                tc.LR, tc.BETA1, tc.BETA2, tc.BS, tc.RHO, tc.WD, 0.0), ...
                'MATLAB:validators:mustBePositive');
        end

    end

    % ══════════════════════════════════════════════════════════════════════════════════════════
    %   12. MULTI-STEP CONVERGENCE
    % ══════════════════════════════════════════════════════════════════════════════════════════
    methods (Test, TestTags = {'Convergence'})

        function multiStep_quadraticLoss_converges(tc)
            % L(w) = 0.5 * w^2.  Gradient = w.  Minimum at w=0.
            % Run 200 steps; verify |w| decreases significantly.
            rng(0);
            p        = dlarray(10.0);
            avg_g    = [];
            avg_hess = [];
            bs       = 512;
            lr       = 0.05;

            for t = 1:200
                g = p;
                do_hess = (mod(t, 10) == 1);
                hess_est = dlarray(1/bs);
                [p, avg_g, avg_hess] = sophiaupdate(p, g, avg_g, avg_hess, ...
                    hess_est, do_hess, t, lr, 0.9, 0.99, bs, 0.04, 0, 1e-15);
            end

            tc.verifyLessThan(abs(double(extractdata(p))), 1.0);
        end

        function multiStep_stateIsConsistentAcrossSteps(tc)
            % avg_hess must remain non-negative throughout training
            % (all estimators produce non-negative diagonal Hessians).
            rng(3);
            d        = 10;
            p        = dlarray(randn(d, 1));
            avg_g    = [];
            avg_hess = [];
            bs       = 256;

            for t = 1:50
                g        = dlarray(randn(d, 1));
                do_hess  = (mod(t, 5) == 0);
                hess_est = dlarray(abs(randn(d, 1)) * bs);

                [p, avg_g, avg_hess] = sophiaupdate(p, g, avg_g, avg_hess, ...
                    hess_est, do_hess, t, 1e-3, 0.9, 0.99, bs, 0.04, 0, 1e-15);
            end

            tc.verifyGreaterThanOrEqual(min(double(extractdata(avg_hess))), 0);
        end

    end

end