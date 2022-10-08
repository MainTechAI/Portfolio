function C = digit_classify(testdata)
%  Input: testdata - N x 3 matrix
%
%  Output: C - integer (0,1,...,9)
%
%  What actions does the function perform:
%  1. exclude Z coordinate from test data
%  2. oversample or downsample test data so that it has 222 points
%  3. scale test data to range [0.0, 1.0]
%  4. transform data from 222 x 2 to 1 x 444
%  5. use k-nn to classify test data

% parameters
p_resample = 222;
k = 5;
metric = "manhattan";

% 1
testdata = testdata(:,1:2);

% 2
p = size(testdata,1);
if p < p_resample
    testdata = oversample(testdata, p_resample);
elseif p > p_resample
    testdata = downsample(testdata, p_resample);
end

% 3
testdata = scale(testdata);

% 4
testdata = reshape(testdata.',[],1);

% 5
load("X_train_knn.mat", "X_train_knn");
load("Y_train_knn.mat", "Y_train_knn");
C = knn(Y_train_knn, X_train_knn, testdata, k, metric);

 
