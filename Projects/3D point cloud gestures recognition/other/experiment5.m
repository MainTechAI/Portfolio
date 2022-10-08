%% distance between points
clear all; close all; clc;

load rel_dist.mat; % from unsuccessful_experiment

rel_dist = rel_dist(:,1);
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

%norm(data{1,1}(1,:)-data{1,1}(2,:))
%sqrt(sum(rel_dist{1,1}(2,:).^2))


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
title("Distribution of distances between points. All data.")

figure
histogram(distances)
title("Distribution of distances between points. All data.")

figure
histogram(distances,50)
title("Distribution of distances between points. All data.")

% pos = data{445,1};
% 
% X = pos(:,1);
% Y = pos(:,2);
% Z = pos(:,3);
% figure;
% plot3(X,Y,Z)