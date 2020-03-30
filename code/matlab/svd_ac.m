rng(1)
a = 0; b=1; n = 1000;
[locs,~] = sort(rand(n,1)*(b-a));
distmatrix = squareform(pdist(locs));
kernel1 = 1./(1+distmatrix.^2); 


[U1, S1, V1] = svd(kernel1); 
figure(1)
semilogy(diag(S1), 'linewidth',4);
legend('A'); title('Spectra of Kernel: Same Field Interactions'); set(gca,'FontSize',12)
