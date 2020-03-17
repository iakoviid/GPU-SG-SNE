% Load data
clear;
load 'mnist_train.mat'
k=2000;
ind=randperm(size(train_X, 1));
train_X=train_X(ind(1:k),:);
train_labels=train_labels(ind(1:k));

% Set parameters
no_dims= 2;
initial_dims= 50;
perplexity= 30;

% preproccessing

% Normalize input data
[X] = preproccessing(train_X);


% Compute pairwise distance matrix
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));

% Compute joint probabilities
[P,beta]= d2p(D, perplexity, 1e-5); % compute affinities using fixed perplexity
nneig=60;
idx=knnsearch(X,X,'k',nneig);
P_tilde=sparse(k,k);
for(i =1:k)
    for j=2:nneig
         P_tilde(i,idx(i,j))=exp(-beta(i)*D(i,idx(i,j)));
    end
    P_tilde(i,:)=P_tilde(i,:)/sum(P_tilde(i,:));
end
clear D

   
% Run t-SNE
no_dims = .0001 * randn(k, no_dims);
[ydata]= tsne_compute(P,P_tilde, train_labels, no_dims,k);

no_dims=size(ydata,2);
%Plot Results 
if(no_dims==1)
%stem(train_labels,ydata)
figure();
for i=1:10
    idx=train_labels==i;
    plot(train_labels(idx),ydata(idx),"*",'linewidth',4);
    hold on;
end
title("MNIST "+n+" sampled points exact t-SNE");
xlabel('digit');
ylabel('value');
end
if(no_dims==2)
    figure();
    gscatter(ydata(:,1),ydata(:,2),train_labels);
    title("MNIST "+n+" sampled points approx t-SNE");

    figure();
    gscatter(ydataex(:,1),ydataex(:,2),train_labels);
    title("MNIST "+n+" sampled points exact t-SNE");

end
    
