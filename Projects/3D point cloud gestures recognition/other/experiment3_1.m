clear all; close all; clc;

% This script creates new file 'data_reduced_222.mat'
% Generating more points up until 222

load("./data/data.mat");
%data = data_reduced;

s = 200;
x=data{s+1,1};
X1 = x(:,1);
Y1 = x(:,2);
Z1 = x(:,3);

x=data{s+2,1};
X2 = x(:,1);
Y2 = x(:,2);
Z2 = x(:,3);

x=data{s+3,1};
X3 = x(:,1);
Y3 = x(:,2);
Z3 = x(:,3);

figure;
plot3(X1,Y1,Z1);
hold on;
plot3(X2,Y2,Z2);
plot3(X3,Y3,Z3);
%%
points = 222;

for i = 1:size(data,1)
    coords = data{i,1};
    if size(coords,1)<points
        data{i,1} = oversample(coords,points);
    elseif size(coords,1)>points
        data{i,1} = downsample(coords,points);
    end
end

x=data{s+1,1};
X1 = x(:,1);
Y1 = x(:,2);

x=data{s+2,1};
X2 = x(:,1);
Y2 = x(:,2);

x=data{s+3,1};
X3 = x(:,1);
Y3 = x(:,2);

figure;
plot(X1,Y1)
hold on;
plot(X2,Y2)
plot(X3,Y3)

data_reduced_222 = data;

save('data_reduced_222.mat','data_reduced_222');
