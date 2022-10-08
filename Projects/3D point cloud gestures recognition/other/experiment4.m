clear all; close all; clc;

% This script generates .CSV file for Python test.
% Transform data to the format that is suited for training classifier

load data_reduced_222_scaled.mat;
data = data_reduced_222_scaled;
final_data = transform_data(data);

save('final_data.mat','final_data');
writematrix(final_data,'final_data.csv')

X_train_knn = final_data(:,1:end-1)';
Y_train_knn = final_data(:,end)';
save('X_train_knn.mat','X_train_knn');
save('Y_train_knn.mat','Y_train_knn');
