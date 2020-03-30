function     [Fatr_tilde,Frep_tilde]=aproxGradient(ydata,P,no_dims,n)
    % Compute joint probability that point i and j are neighbors
%     sum_ydata = sum(ydata .^ 2, 2);
%     num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
%     num(1:n+1:end) = 0;                                                 % set diagonal to zero
%     
%     PP=(P.*num);
%     
%     Fatr_tilde=4*(diag(sum(PP, 1)) - PP) * ydata;
 Fatr_tilde=attractiveSparse(ydata,P);
Frep_tilde=4*repulsive(ydata,n,no_dims);  



end

