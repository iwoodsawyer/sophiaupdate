%% Language Translation Example with SOPHIA Implementation
% This example shows how to train a German to English language translator
% using a recurrent sequence-to-sequence encoder-decoder model with
% attention, optimized with the SOPHIA optimizer (GNB Hessian).
%
% Key differences from original:
% 1. modelLossWithHessian samples labels from network predictions
% 2. Hessian is computed from gradients of sampled loss (GNB method)
% 3. Hessian is scaled by total token count (batch_size * sequence_length)
% 4. Training loop passes numTokens to sophiaupdate, not miniBatchSize
%

% Detect execution environment
executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    env = "gpu";
else
    env = "cpu";
end

%% Load Training Data

% Download and extract the English-German Tab-delimited Bilingual Sentence Pairs data set.
downloadFolder = tempdir;
url = "https://www.manythings.org/anki/deu-eng.zip";
filename = fullfile(downloadFolder,"deu-eng.zip");
dataFolder = fullfile(downloadFolder,"deu-eng");

if ~exist(dataFolder,"dir")
    fprintf("Downloading English-German Tab-delimited Bilingual Sentence Pairs data set (7.6 MB)... ")
    websave(filename,url);
    unzip(filename,dataFolder);
    fprintf("Done.\n")
end

% Create a table that contains the sentence pairs specified as strings
filename = fullfile(dataFolder,"deu.txt");

opts = delimitedTextImportOptions(...
    Delimiter="\t", ...
    VariableNames=["Target" "Source" "License"], ...
    SelectedVariableNames=["Source" "Target"], ...
    VariableTypes=["string" "string" "string"], ...
    Encoding="UTF-8");

% View the first few sentence pairs in the data.
data = readtable(filename, opts);
head(data)

% View the number of remaining observations.
discardProp = 0.70;
idx = size(data,1) - floor(discardProp*size(data,1)) + 1;
data(idx:end,:) = [];
size(data,1)

% Split the data into training and test partitions containing 90% and 10%
% of the data, respectively.
trainingProp = 0.9;
idx = randperm(size(data,1),floor(trainingProp*size(data,1)));
dataTrain = data(idx,:);
dataTest = data;
dataTest(idx,:) = [];

head(dataTrain)

% View the number of training observations.
numObservationsTrain = size(dataTrain,1)


%% Preprocess Data

% Preprocess the text data using the preprocessText function
documentsGerman = preprocessText(dataTrain.Source);
encGerman = wordEncoding(documentsGerman);

% Convert the target data to sequences using the same steps.
documentsEnglish = preprocessText(dataTrain.Target);
encEnglish = wordEncoding(documentsEnglish);

% View the vocabulary sizes of the source and target encodings.
numWordsGerman = encGerman.NumWords
numWordsEnglish = encEnglish.NumWords


%% Define Encoder and Decoder Networks

% Create the encoder and decoder networks
embeddingDimension = 128;
numHiddenUnits = 128;

[lgraphEncoder,lgraphDecoder] = languageTranslationLayers(...
    embeddingDimension,numHiddenUnits,numWordsGerman,numWordsEnglish);

netEncoder = dlnetwork(lgraphEncoder);
netDecoder = dlnetwork(lgraphDecoder);
netDecoder.OutputNames = ["softmax" "context" "lstm2/hidden" "lstm2/cell"];



%% Specify Training Options with SOPHIA Settings

% Initialize mini-batch size, epochs and learning rate
miniBatchSize = 64;
numEpochs = 15;

% Initialize the options for Adam optimization.
learnRate = 0.001;
gradientDecayFactor = 0.965;
hessianDecayFactor = 0.99;
batchSize = miniBatchSize;
clippingThreshold = 0.04;
hessianEstimateFreq = 10;

% Train using gradually decaying values
epsilonStart = 0.5;
epsilonEnd = 0;

% Sort the training sequences by sequence length.
sequenceLengths = doclength(documentsGerman);
[~,idx] = sort(sequenceLengths);
documentsGerman = documentsGerman(idx);
documentsEnglish = documentsEnglish(idx);


%% Train Model with SOPHIA Optimizer
% Train the model using a custom training loop.
% Key changes in this section:
% 1. modelLossWithHessian now takes miniBatchSize and sequenceLength
% 2. numTokens computed and passed to sophiaupdate
% 3. Hessian properly scaled and sampled

% Create array datastores for the source and target data
adsSource = arrayDatastore(documentsGerman);
adsTarget = arrayDatastore(documentsEnglish);
cds = combine(adsSource,adsTarget);

% Create a mini-batch queue to automatically prepare mini-batches for training.
mbq = minibatchqueue(cds, 4, ...
    "OutputEnvironment", env, ... % Move data to GPU automatically
    "OutputCast","single",...
    "MiniBatchSize", miniBatchSize, ...
    "MiniBatchFcn", @(X,Y) preprocessMiniBatch(X,Y,encGerman,encEnglish), ...
    "MiniBatchFormat", ["CTB" "CTB" "CTB" "CTB"], ...
    "PartialMiniBatch", "discard");

% Initialize the training progress plot.
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
xlabel("Iteration")
ylabel("Loss")
ylim([0 inf])
grid on

% For the encoder and decoder networks, initialize the values for Sophia optimization.
trailingAvgGencoder = [];
trailingAvgHencoder = [];
trailingAvgGdecoder = [];
trailingAvgHdecoder = [];

% Create an array of values for scheduled sampling.
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
numIterations = numIterationsPerEpoch * numEpochs;
epsilon = linspace(epsilonStart,epsilonEnd,numIterations);

% Loop over epochs.
iteration = 0;
iteration_hessian = 0;
start = tic;
lossMin = inf;
reset(mbq)

for epoch = 1:numEpochs

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T,maskT,decoderInput] = next(mbq);

        % Compute total token count for proper Hessian scaling
        [~, miniBatchSize_actual, sequenceLength_actual] = size(X);
        numTokens = miniBatchSize_actual * sequenceLength_actual;

        % Compute loss and gradients.
        hessEstEncoder = [];
        hessEstDecoder = [];
        doHessian = (iteration == 1) || (mod(iteration, hessianEstimateFreq) == 0);
        if doHessian
            iteration_hessian = iteration_hessian + 1;
            % With GNB sampling
            [loss,gradientsEncoder,gradientsDecoder,hessEstEncoder,...
                hessEstDecoder,YPred] = dlfeval(@modelLossWithHessian,...
                netEncoder,netDecoder,X,T,maskT,decoderInput,...
                epsilon(iteration),miniBatchSize_actual,...
                sequenceLength_actual);
        else
            % Without Hessian (every non-estimation iteration)
            [loss,gradientsEncoder,gradientsDecoder,YPred] = dlfeval(...
                @modelLoss,netEncoder,netDecoder,X,T,maskT,...
                decoderInput,epsilon(iteration));
        end

        % Update network learnable parameters using sophiaupdate.
        [netEncoder, trailingAvgGencoder, trailingAvgHencoder] = sophiaupdate(...
            netEncoder, gradientsEncoder, trailingAvgGencoder, ...
            trailingAvgHencoder, hessEstEncoder, doHessian, iteration, ...
            iteration_hessian, learnRate, gradientDecayFactor, ...
            hessianDecayFactor, clippingThreshold, numTokens);

        [netDecoder, trailingAvgGdecoder, trailingAvgHdecoder] = sophiaupdate(...
            netDecoder, gradientsDecoder, trailingAvgGdecoder, ...
            trailingAvgHdecoder, hessEstDecoder, doHessian, iteration, ...
            iteration_hessian, learnRate, gradientDecayFactor, ...
            hessianDecayFactor, clippingThreshold, numTokens);

       
        if iteration == 1 || mod(iteration,10) == 0
             % Generate translation for plot.
            strGerman = ind2str(X(:,1,:),encGerman);
            strEnglish = ind2str(T(:,1,:),encEnglish,Mask=maskT);
            strTranslated = ind2str(YPred(:,1,:),encEnglish);
        
            % Display training progress.
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            loss = single(gather(extractdata(loss)));
            addpoints(lineLossTrain,iteration,loss)

            hessianMarker = "";
            if doHessian
                hessianMarker = " [Hessian Est. - GNB]";
            end
            title( ...
                "Epoch: " + epoch + ", Iteration: " + iteration + ...
                ", Elapsed: " + string(D) + hessianMarker + newline + ...
                "Source: " + strGerman + newline + ...
                "Target: " + strEnglish + newline + ...
                "Training Translation: " + strTranslated)
            drawnow limitrate

            % Save best network.
            if loss < lossMin
                lossMin = loss;
                netBest.netEncoder = netEncoder;
                netBest.netDecoder = netDecoder;
                netBest.loss = loss;
                netBest.iteration = iteration;
                netBest.D = D;
            end
        end
    end

    % Shuffle.
    shuffle(mbq);
end

% Add the word encodings to the netBest structure
netBest.encGerman = encGerman;
netBest.encEnglish = encEnglish;

% Save the structure in a MAT file.
D = datetime("now",Format="yyyy_MM_dd__HH_mm_ss");
filename = "net_best__sophia__corrected__" + string(D) + ".mat";
save(filename,"netBest");

% Extract the best network from netBest.
netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;


%% Test Model

% Translate the test data using the translateText function listed at the end of the example.
strTranslatedTest = translateText(netEncoder,netDecoder,encGerman,...
    encEnglish,dataTest.Source);

% View a random selection of the test source text, target text, and predicted translations in a table.
numObservationsTest = size(dataTest,1);
idx = randperm(numObservationsTest,8);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)

% Determine the length of the shortest candidate document.
candidates = preprocessText(strTranslatedTest,StartToken="",StopToken="");
references = preprocessText(dataTest.Target,StartToken="",StopToken="");
minLength = min([doclength(candidates); doclength(references)])
if minLength < 4
    ngramWeights = ones(1,minLength) / minLength;
else
    ngramWeights = [0.25 0.25 0.25 0.25];
end

% Calculate the BLEU evaluation scores by iterating over the translations
% and using the bleuEvaluationScore function.
for i = 1:numObservationsTest
    score(i) = bleuEvaluationScore(candidates(i),references(i),...
        NgramWeights=ngramWeights);
end

% Visualize the BLEU evaluation scores in a histogram.
figure
histogram(score);
title("BLEU Evaluation Scores")
xlabel("Score")
ylabel("Frequency")

% View a table of some of the best translations.
[~,idxSorted] = sort(score,"descend");
idx = idxSorted(1:8);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)

idx = idxSorted(end-7:end);
tbl = table;
tbl.Source = dataTest.Source(idx);
tbl.Target = dataTest.Target(idx);
tbl.Translated = strTranslatedTest(idx)


%% Generate Translations

% Generate translations for new data using the translateText function.
strGermanNew = [
    "Wie geht es Dir heute?"
    "Wie heißen Sie?"
    "Das Wetter ist heute gut."];

% Translate the text using the translateText, function listed at the end of the example.
strTranslatedNew = translateText(netEncoder,netDecoder,encGerman,...
    encEnglish,strGermanNew)


%% Prediction Functions

%%% Beam Search Function
% Beam search is technique for exploring different combinations of
% time-step predictions to help find the best prediction sequence.

function str = beamSearch(X,netEncoder,netDecoder,encEnglish,args)

% Parse input arguments.
arguments
    X
    netEncoder
    netDecoder
    encEnglish

    args.BeamIndex = 3;
    args.MaxNumWords = 10;
end

beamIndex = args.BeamIndex;
maxNumWords = args.MaxNumWords;
startToken = "<start>";
stopToken = "<stop>";

% Encoder predictions.
[Z, hiddenState, cellState] = predict(netEncoder,X);

% Initialize context.
miniBatchSize = size(X,2);
numHiddenUnits = size(Z,1);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);
context = dlarray(context,"CB");

% Initialize candidates.
candidates = struct;
candidates.Words = startToken;
candidates.Score = 0;
candidates.StopFlag = false;
candidates.HiddenState = hiddenState;
candidates.CellState = cellState;

% Loop over words.
t = 0;
while t < maxNumWords
    t = t + 1;

    candidatesNew = [];

    % Loop over candidates.
    for i = 1:numel(candidates)

        % Stop generating when stop token is predicted.
        if candidates(i).StopFlag
            continue
        end

        % Candidate details.
        words = candidates(i).Words;
        score = candidates(i).Score;
        hiddenState = candidates(i).HiddenState;
        cellState = candidates(i).CellState;

        % Predict next token.
        decoderInput = word2ind(encEnglish,words(end));
        decoderInput = dlarray(decoderInput,"CBT");

        [YPred,context,hiddenState,cellState] = predict(netDecoder,decoderInput,hiddenState,cellState,context,Z, ...
            Outputs=["softmax" "context" "lstm2/hidden" "lstm2/cell"]);

        % Find top predictions.
        [scoresTop,idxTop] = maxk(extractdata(YPred),beamIndex);
        idxTop = gather(idxTop);

        % Loop over top predictions.
        for j = 1:beamIndex
            candidate = struct;

            % Determine candidate word and score.
            candidateWord = ind2word(encEnglish,idxTop(j));
            candidateScore = scoresTop(j);

            % Set stop translating flag.
            if candidateWord == stopToken
                candidate.StopFlag = true;
            else
                candidate.StopFlag = false;
            end

            % Update candidate details.
            candidate.Words = [words candidateWord];
            candidate.Score = score + log(candidateScore);
            candidate.HiddenState = hiddenState;
            candidate.CellState = cellState;

            % Add to new candidates.
            candidatesNew = [candidatesNew candidate];
        end
    end

    % Get top candidates.
    [~,idx] = maxk([candidatesNew.Score],beamIndex);
    candidates = candidatesNew(idx);

    % Stop predicting when all candidates have stop token.
    if all([candidates.StopFlag])
        break
    end
end

% Get top candidate.
words = candidates(1).Words;

% Convert to string scalar.
words(ismember(words,[startToken stopToken])) = [];
str = join(words);

end

%%% Translate Text Function
% The translateText function takes as input the encoder and decoder
% networks, an input string, and source and target word encodings and
% returns the translated text.

function strTranslated = translateText(netEncoder,netDecoder,encGerman,encEnglish,strGerman,args)

% Parse input arguments.
arguments
    netEncoder
    netDecoder
    encGerman
    encEnglish
    strGerman

    args.BeamIndex = 3;
end

beamIndex = args.BeamIndex;

% Preprocess text.
documentsGerman = preprocessText(strGerman);
X = preprocessPredictors(documentsGerman,encGerman);
X = dlarray(X,"CTB");

% Loop over observations.
numObservations = numel(strGerman);
strTranslated = strings(numObservations,1);
for n = 1:numObservations

    % Translate text.
    strTranslated(n) = beamSearch(X(:,n,:),netEncoder,netDecoder,encEnglish,BeamIndex=beamIndex);
end

end


%% Helper Functions

%%% Sample Targets from Predictions (Vectorized & GPU Compatible)
function [T_sampled, mask] = sampleTargetsFromPredictions(Y, maskT)

% Extract raw data (numeric or gpuArray) from dlarray
% This is necessary because cumsum/sampling are not differentiable
Y_data = extractdata(Y);

% 1. Vectorized Softmax on raw data
Y_shifted = Y_data - max(Y_data, [], 1);
probs = exp(Y_shifted) ./ sum(exp(Y_shifted), 1);

[numClasses, miniBatchSize, sequenceLength] = size(probs);

% 2. Vectorized Categorical Sampling
% Calculate cumulative probabilities along the class dimension (dim 1)
cumsum_p = cumsum(probs, 1);

% Generate random thresholds
r = rand(1, miniBatchSize, sequenceLength, "like", probs);

% Find the first index where cumsum_p >= r
% In vectorized form: count how many classes are < r and add 1
T_sampled = sum(cumsum_p < r, 1) + 1;

% 3. Boundary Check (ensure indices are within 1 to numClasses)
T_sampled = min(T_sampled, numClasses);

mask = maskT;
% Note: T_sampled is returned as a numeric/gpuArray for use as indices
end


%% Model Functions

%%% Model Loss with Hessian (GNB Method)
function [loss,gradientsE,gradientsD,hessianE,hessianD,YPred] = ...
    modelLossWithHessian(netEncoder,netDecoder,X,T,maskT,...
    decoderInput,epsilon,miniBatchSize,sequenceLength)

[Z, hiddenState, cellState] = forward(netEncoder, X);

Y = decoderPredictions(netDecoder, Z, T, hiddenState, cellState, ...
    decoderInput, epsilon);

% Training loss on actual labels
loss = sparseCrossEntropy(Y, T, maskT);
[gradientsE, gradientsD] = dlgradient(loss, ...
    netEncoder.Learnables, netDecoder.Learnables);

% Step 1: Sample labels from network predictions
[T_sampled_raw, mask_sampled] = sampleTargetsFromPredictions(Y, maskT);

% Step 2: Compute SAMPLED loss (different from training loss)
loss_sampled = sparseCrossEntropy_GNB(Y, T_sampled_raw, mask_sampled);

% Step 3: Compute gradients w.r.t. SAMPLED loss (not actual loss)
[gradients_sampled_E, gradients_sampled_D] = dlgradient(...
    loss_sampled, netEncoder.Learnables, netDecoder.Learnables);

% Step 4: Estimate Hessian from sampled gradients (GNB formula)
hessianE = dlupdate(@(g) g.^2, gradients_sampled_E);
hessianD = dlupdate(@(g) g.^2, gradients_sampled_D);

% Decode predictions for visualization
YPred = onehotdecode(Y, 1:size(Y,1), 1, "single");
end


%%% Model Loss Function
% The modelLoss function takes as input the encoder network, decoder
% network, mini-batches of predictors X, targets T, padding mask
% corresponding to the targets maskT, and ϵ value for scheduled sampling.
% The function returns the loss, the gradients of the loss with respect to
% the learnable parameters in the networks gradientsE and gradientsD, and
% the decoder predictions YPred encoded as sequences of one-hot vectors.

function [loss,gradientsE,gradientsD,YPred] = modelLoss(netEncoder,...
    netDecoder,X,T,maskT,decoderInput,epsilon)

% Forward through encoder.
[Z, hiddenState, cellState] = forward(netEncoder,X);

% Decoder output.
Y = decoderPredictions(netDecoder,Z,T,hiddenState,cellState,...
    decoderInput,epsilon);

% Sparse cross-entropy loss.
loss = sparseCrossEntropy(Y,T,maskT);

% Update gradients.
[gradientsE,gradientsD] = dlgradient(loss,netEncoder.Learnables,...
    netDecoder.Learnables);

% For plotting example translations, return the decoder output.
YPred = onehotdecode(Y,1:size(Y,1),1,"single");
end

%%% Decoder Predictions Function
% The decoderPredictions function takes as input, the decoder network, the
% encoder output Z, the targets T, the decoder input hidden and cell state
% values, and the ϵ value for scheduled sampling.

function Y = decoderPredictions(netDecoder,Z,T,hiddenState,cellState,...
    decoderInput,epsilon)

% Initialize context.
numHiddenUnits = size(Z,1);
miniBatchSize = size(Z,2);
context = zeros([numHiddenUnits miniBatchSize],"like",Z);
context = dlarray(context,"CB");

% Initialize output.
idx = (netDecoder.Learnables.Layer == "fc" & ...
    netDecoder.Learnables.Parameter=="Bias");
numClasses = numel(netDecoder.Learnables.Value{idx});
sequenceLength = size(T,3);
Y = zeros([numClasses miniBatchSize sequenceLength],"like",Z);
Y = dlarray(Y,"CBT");

% Forward start token through decoder.
[Y(:,:,1),context,hiddenState,cellState] = forward(netDecoder,...
    decoderInput,hiddenState,cellState,context,Z);

% Loop over remaining time steps.
for t = 2:sequenceLength
    % Scheduled sampling. 

    % Randomly select previous target or previous prediction.
    if rand < epsilon
        % Use target value.
        decoderInput = T(:,:,t-1);
    else
        % Use previous prediction.
        [~,Yhat] = max(Y(:,:,t-1),[],1);
        decoderInput = Yhat;
    end

    % Forward through decoder.
    [Y(:,:,t),context,hiddenState,cellState] = forward(netDecoder,...
        decoderInput,hiddenState,cellState,context,Z);
end
end

%%% Sparse Cross-Entropy Loss for GNB (Vectorized & GPU Compatible)
function loss = sparseCrossEntropy_GNB(Y, T_sampled_indices, maskT)
% Y: [numClasses, miniBatchSize, sequenceLength]
% T_sampled_indices: [1, miniBatchSize, sequenceLength]
% maskT: [1, miniBatchSize, sequenceLength]

[numClasses, miniBatchSize, sequenceLength] = size(Y);

% 1. Robustly extract indices and mask data
% We need raw numeric/gpuArray values for indexing logic
if isdlarray(T_sampled_indices)
    T_data = extractdata(T_sampled_indices);
else
    T_data = T_sampled_indices;
end
if isdlarray(maskT)
    mask_data = extractdata(maskT);
else
    mask_data = maskT;
end

% 2. Numerical Stability: Compute Log-Sum-Exp (keep as dlarray)
maxY = max(Y, [], 1);
logSumExp = maxY + log(sum(exp(Y - maxY), 1));

% 3. Extract Logits for Sampled Classes using Linear Indexing
T_flat = reshape(T_data, 1, []);
numElements = miniBatchSize * sequenceLength;

% Calculate linear offsets to jump to the correct batch/time-step column
offsets = cast((0:numElements-1) * numClasses, 'uint32');
if isa(Y, 'gpuArray')
    offsets = gpuArray(offsets);
end
linearIdx = uint32(T_flat) + offsets;

% Flatten Y and extract specific logits (preserves dlarray gradients)
Y_flat = reshape(Y, [], 1);
sampledLogits = Y_flat(linearIdx);
sampledLogits = reshape(sampledLogits, 1, miniBatchSize, sequenceLength);

% 4. Compute Negative Log Likelihood: -(logit - logSumExp)
loss_all = logSumExp - sampledLogits;

% 4. Apply Mask and Average
loss_all = loss_all .* mask_data;
numValidTokens = sum(mask_data, "all");
loss = sum(loss_all, "all") / numValidTokens;
end


%%% Sparse Cross-Entropy Loss
% The sparseCrossEntropy function calculates the cross-entropy loss between
% the predictions Y and targets T with the target mask maskT, where Y is an
% array of probabilities and T is encoded as a sequence of integer values.

function loss = sparseCrossEntropy(Y,T,maskT)
[numClasses, miniBatchSize, sequenceLength] = size(Y);

% To prevent calculating log of 0, bound away from zero.
precision = underlyingType(Y);
Y = max(Y, eps(precision));

% Vectorized indexing: Extract the logit corresponding to the target class
% Y is [numClasses, miniBatchSize, seqLen], T is [1, miniBatchSize, seqLen]
T_indices = reshape(extractdata(T), 1, []);
offsets = (0:miniBatchSize*sequenceLength-1) * numClasses;
linearIndices = uint32(T_indices + offsets);

Y_flat = reshape(Y, [], 1);
loss = -log(Y_flat(linearIndices));

% Reshape and apply mask
loss = reshape(loss, miniBatchSize, sequenceLength);
loss = loss .* squeeze(maskT);
numValidTokens = sum(maskT, 'all');
loss = sum(loss, 'all') / numValidTokens;
end

%%% Text Preprocessing Function
% The preprocessText function preprocesses the input text for translation
% by converting the text to lowercase, adding start and stop tokens, and
% tokenizing.

function documents = preprocessText(str,args)

arguments
    str
    args.StartToken = "<start>";
    args.StopToken = "<stop>";
end

startToken = args.StartToken;
stopToken = args.StopToken;

str = lower(str);
str = startToken + str + stopToken;
documents = tokenizedDocument(str,CustomTokens=[startToken stopToken]);
end

%%% Mini-Batch Preprocessing Function
% The preprocessMiniBatch function preprocesses tokenized documents for
% training. The function encodes mini-batches of documents as sequences of
% numeric indices and pads the sequences to have the same length.

function [XSource,XTarget,mask,decoderInput] = preprocessMiniBatch(...
    dataSource,dataTarget,encGerman,encEnglish)

documentsGerman = cat(1,dataSource{:});
XSource = preprocessPredictors(documentsGerman,encGerman);

documentsEnglish = cat(1,dataTarget{:});
sequencesTarget = doc2sequence(encEnglish,documentsEnglish,...
    PaddingDirection="none");

[XTarget,mask] = padsequences(sequencesTarget,2,PaddingValue=1);

decoderInput = XTarget(:,1,:);
XTarget(:,1,:) = [];
mask(:,1,:) = [];
end

%%% Predictors Preprocessing Function
% The preprocessPredictors function preprocesses source documents for
% training or prediction. The function encodes an array of tokenized
% documents as sequences of numeric indices.

function XSource = preprocessPredictors(documentsGerman,encGerman)

sequencesSource = doc2sequence(encGerman,documentsGerman,...
    PaddingDirection="none");
XSource = padsequences(sequencesSource,2);
end
