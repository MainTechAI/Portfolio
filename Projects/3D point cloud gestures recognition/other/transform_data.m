function new = transform_data(data) 
    x = data;
    samples = size(x,1);
    [r c] = size(data{1});

    new = zeros(samples, r*c+1);

    for i = 1:samples
       new(i,:) = [ reshape(data{i,1}.',1,[]) data{i,2} ];
    end    