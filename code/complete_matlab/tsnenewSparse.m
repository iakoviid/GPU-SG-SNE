% Load data
clear;
load 'mnist_train.mat'
n=30000;
max_iter=1000;
ind=randperm(size(train_X, 1));
train_X=train_X(ind(1:n),:);
train_labels=train_labels(ind(1:n));

% Set parameters
no_dims= 2;
initial_dims= 50;
perplexity= 30;

% preproccessing
[X] = preproccessing(train_X,initial_dims);


% Compute pairwise distance matrix

nneig=90;
[idx,D]=knnsearch(X,X,'k',nneig);


% Compute joint probabilities
[ beta] = d2pSparse(D, perplexity,1e-5);
P_tilde=sparse(n,n);
for(i=1:n)
    P_tilde(i,idx(i,2:end))=exp(-D(i,2:end)*beta(i));
    s=sum(exp(-D(i,2:end)*beta(i)));
    P_tilde(i,:)=P_tilde(i,:)/s;
end
clear D idx X
   
% Run t-SNE
no_dims = .0001 * randn(k, no_dims);
[ydata]= tsne_computeAprox(P_tilde, train_labels, no_dims,n,max_iter);
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


end
if(no_dims==3)
    graph3D(ydata,train_labels);    
end
    
