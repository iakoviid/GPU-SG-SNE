A = magic(4);
A(:,4) = [];
[Arow Acol] = size(A);
[irow icol] = ndgrid((0:Arow-1),(0:Acol-1));  % block indices
m0 = 3; n0 = 5;   % these are the row and column indices
                  %  of the upper left corner of the block
% for loop here to index m0,n0, and A
S = sparse(m0+irow(:),n0+icol(:),A(:))
% end
%full(S)
%A