classdef luongAttentionLayer < nnet.layer.Layer & nnet.layer.Formattable

    properties
        NumHiddenUnits
    end

    properties (Learnable)
        % Layer learnable parameters.

        Weights
    end
    
    methods
        function layer = luongAttentionLayer(numHiddenUnits,args)
            % Create an Luong attention layer.

            arguments
                numHiddenUnits
                args.Name = "";
            end
            layer.Name = args.Name;
            layer.Description = "Luong Attention";
            
            layer.InputNames = ["hidden" "encoder"];
            
            layer.NumHiddenUnits = numHiddenUnits;
            
            sz = [numHiddenUnits numHiddenUnits];
            numOut = numHiddenUnits;
            numIn = numHiddenUnits;
            layer.Weights = initializeGlorot(sz,numOut,numIn);
        end
        
        function Z = predict(layer, X1, X2)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer  - Layer to forward propagate through
            %         X1, X2 - Input data
            % Outputs:
            %         Z      - Output of layer forward function

            hiddenState = X1;
            encoderOutputs = X2;
            weights = layer.Weights;
            
            % Initialize attention energies.
            [miniBatchSize, sequenceLength] = size(encoderOutputs, 2:3);
            attentionEnergies = zeros([sequenceLength miniBatchSize],"like",hiddenState);

            % Attention energies.
            encoderOutputs = stripdims(encoderOutputs);
            hWX = hiddenState .* pagemtimes(weights,encoderOutputs);
            for tt = 1:sequenceLength
                attentionEnergies(tt,:) = sum(hWX(:,:,tt),1);
            end

            % Attention scores.
            attentionScores = softmax(attentionEnergies,DataFormat="CB");

            % Context.
            encoderOutputs = permute(encoderOutputs, [1 3 2]);
            attentionScores = permute(attentionScores,[1 3 2]);
            context = pagemtimes(encoderOutputs,attentionScores);
            context = squeeze(context);
            
            Z = dlarray(context,"CB");
        end
    end
end