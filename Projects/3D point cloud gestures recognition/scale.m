function coords = scale(x)
    % Input:
    % x - (N x m) matrix [Integer,Real]
    %   N - number of points 
    %   m - dimensionality of the space in which points are located

    % Output:
    % coords - (N x m) matrix [Real]. Points after scaling
    
    % What actions does the function perform:
    % 1. find m maximum values
    % 2. find m minimum values
    % 3. scale coordinates to [0.0, 1.0]
    
    max_c = max(x);
    min_c = min(x);

    x = (x - min_c) ./ (max_c - min_c);
    
    coords = x;