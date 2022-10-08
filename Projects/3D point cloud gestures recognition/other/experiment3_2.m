clear all; close all; clc;

% This script changes data!
% Scaling data:
% Since the real size of the digit doesn't really matter we can scale the
% data

load("./data/data_reduced_222.mat");
data = data_reduced_222;

x=data{1,1};
X1 = x(:,1);
Y1 = x(:,2);

x=data{2,1};
X2 = x(:,1);
Y2 = x(:,2);

x=data{3,1};
X3 = x(:,1);
Y3 = x(:,2);

figure;
plot(X1,Y1)
hold on;
plot(X2,Y2)
plot(X3,Y3)


for i = 1:size(data,1)
    coords = data{i,1};
    data{i,1} = scale(coords);
end

x=data{1,1};
X1 = x(:,1);
Y1 = x(:,2);

x=data{2,1};
X2 = x(:,1);
Y2 = x(:,2);

x=data{3,1};
X3 = x(:,1);
Y3 = x(:,2);

figure;
plot(X1,Y1)
hold on;
plot(X2,Y2)
plot(X3,Y3)

data_reduced_222_scaled = data;

save('./data/data_reduced_222_scaled.mat','data_reduced_222_scaled');

