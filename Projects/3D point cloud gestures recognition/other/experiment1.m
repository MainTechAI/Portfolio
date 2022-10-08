clear all; close all; clc;

% it doesn't affect the data
% it is just a visual exploration

load data.mat;
x = data(:,1);

points_num = zeros(1000,1);

for i = 1:1000
    points_num(i) = size(x{i},1);
end

figure;
plot(points_num)
title("Number of points for each sample")

figure
histogram(points_num,25)
title("Number of points, distribution")

min_p_num = min(points_num)
max_p_num = max(points_num)

% most common number of points [30-50-70]
% min = 19
% max = 222

%% 2 check distance between two points
clear all; close all; clc;

% it doesn't affect the data
% it is just a visual exploration

load data.mat;


rel_dist = data(:,1);
for i = 1:1000
   coords = rel_dist{i,1};
   points = size(coords,1);
   
   rel = zeros(points,3);
   for j = 2:points
       c1 = coords(j-1,:);
       c2 = coords(j,:);
       rel(j,:) = c2-c1;
   end
   rel_dist{i,1} = rel;    
end


distances=[];
for i = 1:1000
   for j = 2:size(rel_dist{i,1},1)
       dist=sqrt(sum(rel_dist{i,1}(j,:).^2));
       distances(end+1)=dist;
       if dist>400
          disp("("+num2str(i)+","+num2str(j)+")="+num2str(dist)) 
       end
   end  
end

mean_dist = mean(distances)

figure;
plot(distances);
hold on;
yline(mean_dist,'r');
title("Distances between points. All data.")

figure
histogram(distances,200)
title("Distribution of distances between points. All data.")

pos = data{445,1};

X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


