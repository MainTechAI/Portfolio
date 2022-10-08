clear all; close all; clc;

data = cell(1000,2);
points_num = zeros(1000,1);
i = 0;

for digit = 0:9
    for n = 1:100
       file = "stroke_" + num2str(digit) + "_0"; 
       if n<10
           file = file + "00"; 
       elseif n>=10 & n<100
           file = file + "0"; 
       end
       file = "training_data/" + file + num2str(n) + ".csv";
       x = load(file);
       
       i = i + 1;
       data{i,1} = x;
       data{i,2} = digit;
    end
end

save('data.mat','data');