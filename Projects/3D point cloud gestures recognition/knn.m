function C = knn(trainclass, traindata, data, k, metric)
    % Input: 
    % 1. trainclass - (1 x 1000) matrix [Real]. Labels of training data
    % 2. traindata  - (444 x 1000) matrix [Real]. Trainig data
    % 3. data - (444 x 1) matrix [Real]. Testing data that will be classified.
    % 4. k - number of nearest neighbours for k-nn algorithm
    % 5. metric - ['manhattan','chebyshev','canberra','euclidean'] metric 
    %   that will be used during classification. Euclidean distance is 
    %   used by default.
    
    % Output:
    % C - [Integer]. Class that was predicted.
    
    % What actions does the function perform:
    % 1. Find all unique values of classes
    % 2. Find number of samples in train and test data
    % 3. Compute distance matrix. The matrix contains distances between
    %   points of testing and training data.
    % 4. Loop through testing data (in case if there is more than 1 example)
    % 5. Get nearest points from the training set
    % 6. Get distances to the k nearest train points
    % 7. Get indexes of these points in training data
    % 8. Using function chooseClass predict to which class the point 
    %   belongs to. The function slightly more sophisticated than just 
    %   choosing the most frequent class. More info in comments.
    
    
    % 1
    CV = unique(trainclass); 
    
    % 2
    n = size(data,2);
    m = size(traindata,2);
    
    % 3
    DM = zeros(n,m);
    if metric=="manhattan"
        for i = 1:n
            for j = 1:m
                DM(i,j) = sum(abs(data(:,i) - traindata(:,j))); 
            end
        end  
    elseif metric=="chebyshev"
        for i = 1:n
            for j = 1:m
                DM(i,j) = max(abs(data(:,i) - traindata(:,j))); 
            end
        end     
    elseif metric=="canberra"
        for i = 1:n
            for j = 1:m 
                DM(i,j) = sum(abs(data(:,i) - traindata(:,j)) ./ ...
                                 (abs(data(:,i))+abs(traindata(:,j)))); 
            end
        end 
    else
        for i = 1:n
            for j = 1:m
                DM(i,j) = norm(data(:,i) - traindata(:,j)); 
            end
        end
    end
    
    % 4
    C = zeros(1,size(data,2));
    counter = 0;
    for i = 1:n
        counter = counter + 1;
        
         % 5
        [D,I] = sort( DM(i,1:m) );
        
        % 6 
        D = D(2:k+1); 
        
        % 7
        I = I(2:k+1);
        NC = trainclass(I); % Nearest Classes

        % 8
        C(counter) = chooseClass(D,NC,CV);
    end
    
    
    function C = chooseClass(D,NC,CV)
        % compute class frequencies using 'tabulate' function
        table = tabulate(NC);
        % we don't need values from 3rd column
        % we will later fill them with different values
        table(:,3) = 0;

        % find most frequent class among k-nearest classes
        [max_C, max_I] = max(table(:,2));

        if sum(table(:,2)==max_C)==1 % if there is single most frequent class
            C = table(max_I,1);
        else % if there is more then one frequent classes
            % first remove less frequent classes
            new_table = deleteNonEqual(table);

            % then for each class compute sum of distances
            for class = CV
                [r, c] = find(new_table(:,1)==class);
                if ~isempty(r) && ~isempty(c)
                    idx = find(NC==class);
                    new_table(r,3) = sum(D(idx));
                end
            end

            % and pick the closest class
            [min_D, min_D_I] = min(new_table(:,3));

            if sum(new_table(:,3)==min_D)==1
                C = new_table(min_D_I,1);
            else % this is very unlikely to happen
                % in case if it happen just choose first value in table
                C = chooseClass(D(1:end-1),NC(1:end-1),CV);
            end
        end
    end


    function table = deleteNonEqual(table)
        % delete the least frequent classes 
        [maxV, ~] = max(table(:,2));
        [r, ~] = find(table(:,2)==maxV);
        table = table(r,:);
    end

end