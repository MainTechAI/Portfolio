function points = oversample(x, max_p)
    % Input: 
    % 1. x - (N x m) matrix [Integer,Real]
    %   N - number of points
    %   m - dimensionality of the space in which points are located
    % 2. max_p - [Integer] number of point that x should cointain after
    % oversampling
    
    % Output:
    % points - (N x max_p) matrix [Integer,Real]. Points after oversampling
    
    % What actions does the function perform:
    % 1. calculate how much points should be generated
    % 2. calculate distance vector
    % 3. check distances and add new points between the points with the
    % biggest distance. Repeat max_p times
    
    
    % 1
    rows = size(x,1);
    n_gen = max_p-rows; 
    
    % 2
    dv = pairwise_dist(x);
    
    % 3
    for i = 1:n_gen
        % find between which points is the greatest distance
        [V,I] = max(dv);
        
        % simply add a new point between these points
        new_p = (x(I,:) + x(I+1,:))/2;
        x = [x(1:I,:); new_p ; x(I+1:end,:)];
        
        % update distance vector
        new_d = V/2;
        dv(I) = new_d;
        dv = [dv(1:I); new_d ; dv(I+1:end)];
    end
    
    points = x;
    
    function d = pairwise_dist(z)
        % calculate euclidean distance between two vectors
        z1 = z(1:end-1,:);
        z2 = z(2:end,:);
        
        % second vector is shifted by one value, in order to calculate
        % pairwise distances
        d = vecnorm(z1-z2, 2, 2);
    end

end