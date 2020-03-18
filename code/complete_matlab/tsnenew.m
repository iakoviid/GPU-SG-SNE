% Load data
clear;
load 'mnist_train.mat'
n=500;
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
sum_X = sum(X .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
nneig=90;
idx=knnsearch(X,X,'k',nneig);

%D=D.*points;

% Compute joint probabilities
[beta]= d2pbeta(D, perplexity, 1e-5); % compute affinities using fixed perplexity
P_tilde=spalloc(n,n,nneig-1);
for(i =1:n)
    for j=2:nneig
         P_tilde(i,idx(i,j))=exp(-beta(i)*D(i,idx(i,j)));
    end
    P_tilde(i,:)=P_tilde(i,:)/sum(P_tilde(i,:));
end
clear D

   
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
    title("MNIST "+n+" sampled points approx t-SNE");

end
    
