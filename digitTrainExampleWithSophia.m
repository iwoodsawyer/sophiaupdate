%% 1. Load and Prepare Data
[XTrain, YTrain] = digitTrain4DArrayData;

% Convert to dlarray and rescale
XTrain = dlarray(single(XTrain)/255, 'SSCB');
classes = categories(YTrain);
numClasses = numel(classes);

%% 2. Define Network
layers = [
    imageInputLayer([28 28 1], 'Normalization', 'none')
    convolution2dLayer(3, 8, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer];

lgraph = layerGraph(layers);
netAdam = dlnetwork(lgraph);
netSophia = dlnetwork(lgraph); % Identical copy for Sophia

%% 3. Training Hyperparameters
numEpochs = 100;
miniBatchSize = 128;
numObservations = size(XTrain, 4);
numIterationsPerEpoch = floor(numObservations/miniBatchSize);

% Adam Params
lrAdam = 1e-3;
avgG_A = [];
avgSq_A = [];

% Sophia Params
lrSophia = 1e-3;
hessInterval = 10;
hessMethod = 'GNB';   % Toggle between 'GNB' and 'Hutchinson'
avgG_S = [];
avgH_S = [];
% bs = tokens per batch. For images, we use TotalPixels
bs = miniBatchSize * 28 * 28; 

%% 4. Training Loop
iteration = 0;
iteration_hessian = 0;
lossAdamLog = [];
lossSophiaLog = [];

% Check GPU availability and select device
gpuDevice;
useGPU = gpuDeviceCount > 0;

% Transfer data to GPU if available
if useGPU
    XTrain = gpuArray(XTrain);
    YTrain = gpuArray(single(YTrain));
end

figure;
lineAdam = animatedline('Color', 'r', 'LineWidth', 1.5);
lineSophia = animatedline('Color', 'b', 'LineWidth', 1.5);
legend('Adam', 'Sophia');
xlabel("Iteration"); ylabel("Loss"); title("Convergence Comparison");
grid on;

for epoch = 1:numEpochs
    idx = randperm(numObservations);
    
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Get Mini-batch
        batchIdx = idx((i-1)*miniBatchSize+1 : i*miniBatchSize);
        X = XTrain(:,:,:,batchIdx);
        Y = dummyvar(YTrain(batchIdx))';
        Y = dlarray(single(Y));
        
        % --- Update Adam ---
        [lossA, gradA] = dlfeval(@modelLoss, netAdam, X, Y);
        [netAdam, avgG_A, avgSq_A] = adamupdate(netAdam, gradA, avgG_A, avgSq_A, iteration, lrAdam);
        
        % --- Update Sophia ---
        % 1. Standard Gradient
        [lossS, gradS] = dlfeval(@modelLoss, netSophia, X, Y);
        
        % 2. Hessian Estimation (GNB Sampling)
        doHess = (mod(iteration, hessInterval) == 0) || (iteration == 1);
        hessEst = [];
        if doHess      
            iteration_hessian = iteration_hessian + 1; 

            if strcmp(hessMethod, 'GNB')
                % Option 1: GNB Sampling (Sample labels from model distribution)
                logits = predict(netSophia, X);
                Y_sampled = sampleCategorical(logits);
                [~, gradSampled] = dlfeval(@modelLoss, netSophia, X, Y_sampled);
                hessEst = dlupdate(@(g) g.*g, gradSampled);
            else
                % Option 2: Hutchinson Estimator (Hessian-vector product)
                u = dlupdate(@(x) randn(size(x), 'like', x), gradS);
                % Compute Hessian-vector product H*u
                hvp = dlfeval(@computeHVP, netSophia, X, Y, u);
                % Hutchinson diagonal estimate: u .* (H*u)
                hessEst = dlupdate(@(a,b) max(a.*b, 0), u, hvp);
            end
        end
        
        [netSophia, avgG_S, avgH_S] = sophiaupdate(netSophia, gradS, avgG_S, avgH_S, ...
            hessEst, doHess, iteration, iteration_hessian, lrSophia,0.9,0.99,0.01,bs);
        
        % Logging
        addpoints(lineAdam, iteration, double(extractdata(lossA)));
        addpoints(lineSophia, iteration, double(extractdata(lossS)));
        drawnow limitrate
    end
end

%% Helper Functions
function [loss, gradients] = modelLoss(net, X, Y)
    probs = forward(net, X);
    loss = crossentropy(probs, Y);
    gradients = dlgradient(loss, net.Learnables);
end

function hvp = computeHVP(net, X, Y, v)
    % Forward pass and first gradient
    probs = forward(net, X);
    loss = crossentropy(probs, Y);
    grad = dlgradient(loss, net.Learnables, 'RetainData', true);
    
    % Differentiable dot product: sum over all learnables of grad .* v
    gvTable = dlupdate(@(g, vi) sum(g .* vi, 'all'), grad, v);
    
    % Sum the per-parameter scalars into a single scalar dlarray
    vals = gvTable.Value;
    gv = vals{1};
    for k = 2:numel(vals)
        gv = gv + vals{k};
    end
    
    % Second gradient gives H*v
    hvp = dlgradient(gv, net.Learnables);
end

function Y_sampled = sampleCategorical(logits)
    % Manually sample from the softmax output
    probs = extractdata(logits);
    numObs = size(probs, 2);
    numClasses = size(probs, 1);
    Y_sampled = zeros(numClasses, numObs, 'single');
    
    for j = 1:numObs
        p = probs(:, j);
        classIdx = randsample(1:numClasses, 1, true, p);
        Y_sampled(classIdx, j) = 1;
    end
    Y_sampled = dlarray(Y_sampled);
end