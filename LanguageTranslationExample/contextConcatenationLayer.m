classdef contextConcatenationLayer < nnet.layer.Layer & nnet.layer.Formattable

    methods
        function layer = contextConcatenationLayer(args)
            % Create a contextConcatenationLayer.

            arguments
                args.Name = "";
            end
            layer.Name = args.Name;
            layer.Type = "Concatenation";
            layer.Description = "Context concatenation";

            layer.InputNames = ["input" "context"];
        end

        function Z = predict(layer, X, context)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X, context  - Input data
            % Outputs:
            %         Z - Output of layer forward function

            % Add "T" dimension label.
            context = dlarray(context,"CBT");
            
            % Concatenate.
            sequenceLength = size(X,3);
            Z = cat(1, X, repmat(context, [1 1 sequenceLength]));            
        end
    end
end