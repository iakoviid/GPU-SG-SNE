function [S] = kerneleval1(points)
distmatrix = squareform(pdist(points));
S = 1./(1+distmatrix.^2);
end

