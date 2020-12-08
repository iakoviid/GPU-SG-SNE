blocksize = 4;
nblocks = 3;
nb=nblocks;
% The elements contained in the various blocks.
% If each block is the same, it is easy to build.
% I've just used random elements.
Z = rand(blocksize,blocksize,nblocks);
[subr,subc] = meshgrid(1:blocksize);
% The pattern matrix:
blockrc = [1 2;2 1;3 3];
rowind = subr + reshape((blockrc(:,1)-1)*blocksize,[1 1 nb]);
colind = subc + reshape((blockrc(:,2)-1)*blocksize,[1 1 nb]);
% create A in one call to sparse
A = sparse(rowind(:),colind(:),Z(:),blocksize*max(blockrc(:,1)),blocksize*max(blockrc(:,2)));

spy(A)