% Load data
% load 'mnist_train.mat'
clear;
load 'mnist_train.mat'
k=7000;
ind=randperm(size(train_X, 1));
train_X=train_X(ind(1:k),:);
train_labels=train_labels(ind(1:k));

% Set parameters
no_dims= 2;
initial_dims= 50;
perplexity= 30;
% Run t−SNE
mappedX=tsne(train_X, [],no_dims,initial_dims,perplexity);
% Plot results
figure();
gscatter(mappedX(:,1),mappedX(:,2),train_labels);