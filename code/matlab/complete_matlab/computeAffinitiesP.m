function [P_tilde]=computeAffinitiesP(X,perplexity,tol,nneig,n)
disp('Computing nearest neighboors');
[idx,D]=knnsearch(X,X,'k',nneig);
[ ~,v] = d2pSparse2(D, perplexity,tol,idx);
a=[];
for i=1:nneig-1
    a=[a [1:n]'];
end
P_tilde=sparse(a,idx(:,2:end),v);

%P_tilde(1:n + 1:end) = 0;                                 % set diagonal to zero
P_tilde = 0.5 * (P_tilde + P_tilde');                                 % symmetrize P-values
P_tilde = (P_tilde ./ sum(P_tilde(:)));                   % make sure P-values sum to one


end

