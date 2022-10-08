clear all;close all;clc;

% it doesn't affect the data
% it is just a visual exploration
% just check some data 0-9, one example per digit

load data.mat;

pos = data{1,1};

X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)



pos = data{101,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{201,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{301,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{401,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{501,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{601,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{701,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{801,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


pos = data{901,1};
X = pos(:,1);
Y = pos(:,2);
Z = pos(:,3);
figure;
plot3(X,Y,Z)


