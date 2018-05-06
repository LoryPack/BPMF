clc;
clear all;
M = dlmread("ratings.csv", ",", R1=1, C1=0);  % "\t" -> separator
M = M(1:end, 1:3); % only first three columns
M = [M(:,1:2), 2.*M(:,3)];
M = M(randperm(size(M,1)), :);
M_train = M(1:80000,:);
S = spconvert(M_train);
mmwrite("ml-latest-small-train.mtx", S);
M_test = M(80001:end,:);
S = spconvert(M_test);
mmwrite("ml-latest-small-test.mtx", S);

