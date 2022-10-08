function points = downsample(x, min_p)
    % Input: 
    % 1. x - (N x m) matrix of type [Integer,Real]
    %   N - number of points 
    %   m - dimensionality of the space in which points are located
    % 2. min_p - [Integer] number of point that x should cointains after
    % downsampling
    
    % Output:
    % points - (N x min_p) matrix of type [Integer,Real]. Points after 
    % downsampling
    
    % What actions does the function perform:
    % 1. use function resample from Matlab's Signal Processing Toolbox 
    
    n = size(x,1);

    points = resample(x,min_p,n);