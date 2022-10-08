function coords = scale(x)
    % this function scales coordinates
    % new values in between [0,1]
    % every sample scales to the same range
    
    max_c = max(x);
    min_c = min(x);

    x = (x - min_c) ./ (max_c - min_c);
    
    coords = x;
