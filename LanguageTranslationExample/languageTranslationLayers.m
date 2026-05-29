function [lgraphE,lgraphD] = languageTranslationLayers(embeddingDimension,numHiddenUnits,numWordsSource,numWordsTarget)

% Encoder.
layers = [
    sequenceInputLayer(1)
    wordEmbeddingLayer(embeddingDimension,numWordsSource)
    lstmLayer(numHiddenUnits,HasStateOutputs=true)];
lgraphE = layerGraph(layers);

% Decoder.
outputSize = numWordsTarget + 1;

% Start by defining the layers from the main branch on the network (from
% the decoder input to the decoder output via the main output of the
% stacked LSTM layers). The context input to the decoder and the updated
% context output by the attention layer is a single vector. To concatenate
% this vector with each time step of the input sequences, use the custom
% layer contextConcatenationLayer.
layers = [
    sequenceInputLayer(1)
    wordEmbeddingLayer(embeddingDimension,numWordsTarget,Name="emb")
    contextConcatenationLayer(Name="cat1")
    lstmLayer(numHiddenUnits,HasStateInputs=true,HasStateOutputs=true,Name="lstm2")
    contextConcatenationLayer(Name="cat2")
    fullyConnectedLayer(outputSize)
    softmaxLayer];

lgraphD = layerGraph(layers);

% Add the inputs for the hidden and cell states output by the encoder
% network. Specify the inputs as feature input layers with size matching
% the number of hidden units of the encoder LSTM layer.
layer = featureInputLayer(numHiddenUnits,Name="hidden");
lgraphD = addLayers(lgraphD,layer);
lgraphD = connectLayers(lgraphD,"hidden","lstm2/hidden");

layer = featureInputLayer(numHiddenUnits,Name="cell");
lgraphD = addLayers(lgraphD,layer);
lgraphD = connectLayers(lgraphD,"cell","lstm2/cell");

% Add the input for the initial context vector. Specify the input as a
% feature input layer with size matching the number of hidden units of the
% encoder LSTM layer.

layer = featureInputLayer(numHiddenUnits,Name="context");
lgraphD = addLayers(lgraphD,layer);
lgraphD = connectLayers(lgraphD,"context","cat1/context");

% Add the attention layer and attach it to the hidden output of the LSTM
% layer and the input of the concatenation layer. The attention layer is
% given by the custom layer luongAttentionLayer attached to this example as a
% supporting file. To access this file, open this example as a live script.

layer = luongAttentionLayer(numHiddenUnits,Name="attention");
lgraphD = addLayers(lgraphD,layer);
lgraphD = connectLayers(lgraphD,"lstm2/hidden","attention/hidden");
lgraphD = connectLayers(lgraphD,"attention","cat2/context");

% Add an input layer the encoder output and connect it to the attention
% layer. Specify the input as a sequence input layer with size matching the
% output size of the encoder. The output size of the encoder matches the
% number of hidden units of the encoder LSTM layers.
layer = sequenceInputLayer(numHiddenUnits,Name="encoder");
lgraphD = addLayers(lgraphD,layer);
lgraphD = connectLayers(lgraphD,"encoder","attention/encoder");

end

