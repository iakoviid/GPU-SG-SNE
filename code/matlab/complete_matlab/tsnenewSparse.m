% Load data
 clear;
 load mnist_train.mat
 n=60000;
 %ind=randperm(size(train_X, 1));
 %train_X=train_X(ind(1:n),:);
 %train_labels=train_labels(ind(1:n));
%clear;
%n=1200;
%mu=[0 0 0];
%sigma = [1 0.5 0.5; 0.5 1 0.5;0.5 0.5 1];
%R = mvnrnd(mu,sigma,200);
%mu = [5 0 0];
%R1 = mvnrnd(mu,sigma,1000);
%mu = [2.5 50 0];
%R2 = mvnrnd(mu,sigma,1000);
%train_X=[R;R1];
%train_labels=[ones(200,1);ones(1000,1)*2];
% Set parameters

max_iter=1000;
no_dims= 2;
initial_dims= 50;
perplexity= 30;
nneig=3*perplexity+1;
exag=12;

% preproccessing
%[X] = preproccessing(train_X,initial_dims);



% Compute joint probabilities
[P_tilde]=computeAffinitiesP(X,perplexity,1e-5,nneig,n);


clear X

% Run t-SNE
no_dims = .0001 * randn(n, no_dims);
[ydata]= tsne_computeAprox(P_tilde, train_labels, no_dims,n,max_iter,exag);

no_dims=size(ydata,2);
%Plot Results 
if(no_dims==1)
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
    
