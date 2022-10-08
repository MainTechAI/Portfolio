%% script for digit_classify.m testing

clear all; close all; clc;

load(".\other\data\data.mat");

examples = [57, 161, 279, 308, 426, 557, 662, 787, 815, 957];
classes = (0:9);

for i = 1:10
    X_test = data{examples(i),1};
    C = digit_classify(X_test);
    C_true = classes(i);
    
    figure;
    plot(X_test(:,1), X_test(:,2));
    title("class - " + num2str(C_true) + ", predicted - " + num2str(C));
end

%%

h = randi([1,1000],1,100);
C=[];C_true=[];
for i = h
    X_test = data{i,1};
    C(end+1) = digit_classify(X_test);
    C_true(end+1) = data{i,2};
end
accuracy = 1-(sum(C~=C_true)/size(C,2))

figure;
C = confusionmat(C_true,C);
confusionchart(C);