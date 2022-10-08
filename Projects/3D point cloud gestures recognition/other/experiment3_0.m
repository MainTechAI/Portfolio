clear all; close all; clc;

load("final_data.mat"); % data was changed, you need to create it again

% chi-square tests for feature ranking
[idx,scores] = fscchi2(final_data(:,1:end-1),final_data(:,end) );
dim = mod(idx,3);
dim(dim==0)=3;

figure;
scatter(1:666,dim);
title('Importance');

figure;
plot(scores(1:3:664));
hold on;
plot(scores(2:3:665));
plot(scores(3:3:666));
legend("X","Y","Z");
title('Importance')


scaled_reduced = [];

%% exclude Z dimension

clear all; close all; clc;
load("data.mat");

for i = 1:1000
    data{i,1} = data{i,1}(:,1:2);
end

data_reduced = data;

save('data_reduced.mat','data_reduced');


