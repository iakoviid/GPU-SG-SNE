function [Fatr,Frep]=exactGradient(ydata,P,n)
    % Compute joint probability that point i and j are neighbors
    sum_ydata = sum(ydata .^ 2, 2);
    num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
    num(1:n+1:end) = 0;                                                 % set diagonal to zero
    Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
    
    PP=(P.*num);
    QQ=Q.*num;
    Fatr=4*(diag(sum(PP, 1)) - PP) * ydata;
    Frep=4*(diag(sum(QQ, 1)) - QQ) * ydata;
    

end

