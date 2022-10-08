clear all; close all; clc;
rng('default');

load("final_data.mat");


X = final_data(:,1:end-1)';
Y = final_data(:,end)';

hpart = cvpartition(Y,'Holdout',0.2,'Stratify',true);
idxTrain = training(hpart);
idxTest = test(hpart);

X_train = X(:,idxTrain);
Y_train = Y(:,idxTrain);

X_test = X(:,idxTest);
Y_test = Y(:,idxTest);

X = X_train;
Y = Y_train;

k_max = 100;
folds = 10;
metrics = ["euclidean","manhattan","chebyshev","canberra"];
num_m = size(metrics,2);

% 10-fold Cross Validation 
cvpart = cvpartition(Y,'Kfold',folds,'Stratify',true);
accuracy = size(num_m,k_max,folds);

for m = 1:num_m
    for k = 1:k_max
        for i = 1:folds
            train = training(cvpart,i);
            test = ~train; 
            C = knn(Y(:,train), X(:,train), X(:,test), k, metrics(m));
            accuracy(m,k,i) = 1-(sum(C~=Y(:,test))/size(Y(:,test),2));
        end 
    end
end

mean_acc = mean(accuracy,3);

%%

figure;
plot(mean_acc(1,:));
hold on;
plot(mean_acc(2,:));
plot(mean_acc(3,:));
plot(mean_acc(4,:));

xlabel('k');
ylabel('accuracy');
legend("euclidean","manhattan","chebyshev","canberra");
grid on;

% boxplots, don't need probably
% v = accuracy(1,:,:);
% v = reshape(v,[k_max,folds])';
% figure;
% boxplot(v);
% hold on;
% plot(mean_acc(4,:));

%%

C = knn(Y_train, X_train, X_test, 5, "manhattan");
accuracy = 1-(sum(C~=Y_test)/size(Y_test,2));
conf = confusionmat(Y_test,C);
figure;
confusionchart(conf);
title("Accuracy: "+num2str(accuracy));



%% hold out

hpart = cvpartition(size(Y,2),'Holdout',0.2);
idxTrain = training(hpart);
idxTest = test(hpart);

X_train = X(:,idxTrain);
Y_train = Y(:,idxTrain);

X_test = X(:,idxTest);
Y_test = Y(:,idxTest);

accuracy = size(1,100);
for k = 1:100
    C = knn(Y_train, X_train, X_test, k);
    accuracy(k) = 1-(sum(C~=Y_test)/size(Y_test,2));
end

figure;
plot(accuracy)
xlabel('k')
ylabel('accuracy')
   