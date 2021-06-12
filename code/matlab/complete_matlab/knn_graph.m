 clear;
 load mnist_train.mat
 n=60000;
 ind=randperm(size(train_X, 1));
 train_X=train_X(ind(1:n),:);
 train_labels=train_labels(ind(1:n));
 perplexity= 30;
 nneig=3*perplexity+1;
 initial_dims= 50;

 [X] = preproccessing(train_X,initial_dims);
 disp('Computing nearest neighboors');
 %[idx,D]=knnsearch(X,X,'k',nneig);
 [P_tilde]=computeAffinitiesP(X,perplexity,1e-5,nneig,n);

