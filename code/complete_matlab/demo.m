% Load data
% load 'mnist_train.mat'
 load 'labls.mat'
 load 'trainX.mat'

% Set parameters
no_dims= 2;
initial_dims= 50;
perplexity= 30;
% Run t−SNE
mappedX=tsne(train_X, [],no_dims,initial_dims,perplexity);
% Plot results
gscatter(mappedX(:,1),mappedX(:,2),train_labels);