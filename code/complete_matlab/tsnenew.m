% Load data
clear;
load 'mnist_train.mat'
k=12000;
ind=randperm(size(train_X, 1));
train_X=train_X(ind(1:k),:);
train_labels=train_labels(ind(1:k));

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
nneig=60;
idx=knnsearch(X,X,'k',nneig);

%D=D.*points;

% Compute joint probabilities
[beta]= d2pbeta(D, perplexity, 1e-5); % compute affinities using fixed perplexity
P_tilde=sparse(k,k);
for(i =1:k)
    for j=2:nneig
         P_tilde(i,idx(i,j))=exp(-beta(i)*D(i,idx(i,j)));
    end
    P_tilde(i,:)=P_tilde(i,:)/sum(P_tilde(i,:));
end
clear D
% nneig=60;
% idx=knnsearch(X,X,'k',nneig);
% points=zeros(k,k);
% %build knn graph make p sparse 
% for(i =1:k)
%     points(i,idx(i,:))=1;
%     points(i,i)=0;
% end
% 
% P_tilde=P.*points;
% clear P
% clear points
% P_tilde=(P_tilde+P_tilde')/(2*nneig);    
% P_tilde=sparse(P_tilde);
   
% Run t-SNE
no_dims = .0001 * randn(k, no_dims);
%[ydata, ydataex] = tsne_compute(P,P_tilde, train_labels, no_dims,k);
[ydata]= tsne_computeAprox(P_tilde, train_labels, no_dims,k);
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
    
