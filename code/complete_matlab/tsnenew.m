% Load data
clear;
load 'mnist_train.mat'
ind=randperm(size(train_X, 1));
k=500;
train_X=train_X(ind(1:k),:);
train_labels=train_labels(ind(1:k));
%train_X=train_X(1:k,:);
%train_labels=train_labels(1:k,:);

% Set parameters
no_dims= 2;
initial_dims= 50;
perplexity= 30;

% preproccessing

% Normalize input data
train_X = train_X - min(train_X(:));
train_X = train_X / max(train_X(:));
train_X = bsxfun(@minus, train_X, mean(train_X, 1));

X=train_X;
% Perform preprocessing using PCA
disp('Preprocessing data using PCA...');
if size(X, 2) < size(X, 1)
    C = X' * X;
else
    C = (1 / size(X, 1)) * (X * X');
end
[M, lambda] = eig(C);
[lambda, ind] = sort(diag(lambda), 'descend');
M = M(:,ind(1:initial_dims));
lambda = lambda(1:initial_dims);
if ~(size(X, 2) < size(X, 1))
    M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
end
X = bsxfun(@minus, X, mean(X, 1)) * M;
clear M lambda ind





% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
[P ,beta]= d2p(D, perplexity, 1e-5);                 % compute affinities using fixed perplexity
clear D
nneig=30;
idx=knnsearch(X,X,'k',nneig);
points=zeros(k,k);
%build knn graph make p sparse 
for(i =1:500)
    points(i,idx(i,:))=1;
    points(i,i)=0;
end

P=P.*points;
P=(P+P')/(2*nneig);    
 
   
% Run t-SNE
%figure(2);
ydata = tsne_compute(P, train_labels, no_dims);
figure();
gscatter(ydata(:,1),ydata(:,2),train_labels);
    
    
