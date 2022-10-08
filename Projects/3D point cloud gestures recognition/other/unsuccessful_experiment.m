clear all; close all; clc;

% This script creates new file
% just an idea, to use relative distances instead of coordinates

load data/data_reduced_110_scaled.mat;
data = data_reduced_110_scaled;
rel_dist = data;


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


% transform absolute coordinates to relative distances

for i = 1:1000
   coords = rel_dist{i,1};
   points = size(coords,1);
   rel = zeros(points,2);
   for j = 2:points
       c1 = coords(j-1,:);
       c2 = coords(j,:);
       rel(j,:) = c2-c1;
   end
   rel_dist{i,1} = rel(2:end,:);    
end


x=rel_dist{1,1};
X1 = x(:,1);
Y1 = x(:,2);

x=rel_dist{2,1};
X2 = x(:,1);
Y2 = x(:,2);

x=rel_dist{3,1};
X3 = x(:,1);
Y3 = x(:,2);

figure;
scatter(X1,Y1)
hold on;
scatter(X2,Y2)
scatter(X3,Y3)


%%

final_data_rel1 = transform_data(rel_dist);

save('final_data_rel1.mat','final_data_rel1');
writematrix(final_data_rel1,'final_data_rel1.csv');


%%


function new = transform_data(data) 
    x = data;
    samples = size(x,1);
    [r c] = size(data{1});

    new = zeros(samples, r*c+1);

    for i = 1:samples
       new(i,:) = [ reshape(data{i,1}.',1,[]) data{i,2} ];
    end    
end









