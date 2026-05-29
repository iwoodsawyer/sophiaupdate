function str = ind2str(X,enc,args)
%%in2str Convert array of indices to string.
% str = ind2str(X,enc) converts the array of indices X to string according
% to the specified word encoding enc.
%
% str = ind2str(X,enc,Name=value) specifies additional options using one or
% more name-value pairs.
%
%   Mask       - Mask indicating which indices to include. 
%                The default is an array of ones.

% Parse input arguments.
arguments
    X
    enc
    
    args.Mask = ones(size(X),"like",X);
end

mask = args.Mask;
startToken = "<start>";
stopToken = "<stop>";

% Gather gpuArray data.
X = gather(X);
mask = gather(mask);

% Extract dlarray data.
if isdlarray(X)
    X = extractdata(X);
    mask = extractdata(mask);
end

% Loop over observations.
miniBatchSize = size(X,2);
for n = 1:miniBatchSize

    % Apply mask.
    Tn = X(:,n,:).*mask(:,n,:);
    Tn = squeeze(Tn);

    % Remove padding.
    Tn(Tn==0) = [];

    % Convert indices to text according to start and stop tokens.
    str(n) = join(ind2word(enc,Tn));
    str(n) = eraseBetween(str(n),textBoundary,startToken,Boundaries="inclusive");
    str(n) = eraseBetween(str(n),stopToken,textBoundary,Boundaries="inclusive");
    str(n) = strip(str(n));
end

end