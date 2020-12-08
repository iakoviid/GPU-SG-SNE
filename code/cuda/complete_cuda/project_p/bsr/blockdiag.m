

tic
N = 10;
M = 1000;
Z = sparse(rand(N,N*M));
Zc = mat2cell(Z,N,repmat(N,1,M));
A = blkdiag(Zc{:});
toc
spy(A)
