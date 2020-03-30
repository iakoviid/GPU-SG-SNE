function [ beta,v] = d2pSparse(X, u, tol,idx)


    if ~exist('u', 'var') || isempty(u)
        u = 15;
    end
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-4; 
    end
    
    % Initialize some variables
    sum_X = sum(X .^ 2, 2);
    n = size(X, 1);                     % number of instances
    beta = ones(n, 1);                  % empty precision vector
    logU = log(u);                      % log of perplexity (= entropy)
    v=zeros(n,size(idx,2)-1);

    % Run over all datapoints
    for i=1:n
        D=sum_X(i)+sum_X(:)-2*X*(X(i,:)');
        if ~rem(i, 500)
            disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
        end
        
        % Set minimum and maximum values for precision
        betamin = -Inf; 
        betamax = Inf;

        % Compute the Gaussian kernel and entropy for the current precision
        [H, vthis] = Hbeta(D, beta(i),i);
        
        % Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while abs(Hdiff) > tol && tries < 50
            
            % If not, increase or decrease precision
            if Hdiff > 0
                betamin = beta(i);
                if isinf(betamax)
                    beta(i) = beta(i) * 2;
                else
                    beta(i) = (beta(i) + betamax) / 2;
                end
            else
                betamax = beta(i);
                if isinf(betamin) 
                    beta(i) = beta(i) / 2;
                else
                    beta(i) = (beta(i) + betamin) / 2;
                end
            end
            
            % Recompute the values
            [H, vthis] = Hbeta(D, beta(i),i);
            Hdiff = H - logU;
            tries = tries + 1;
        end
                v(i,:)=vthis(idx(i,2:end));

    end    
    disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
    disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
    disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end
    


% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H,P] = Hbeta(D, beta,i)
   
   
    P=exp(-D*beta);
    P(i)=0;
    s=sum(P.*D);
    sumP = sum(P);
    H = log(sumP) + beta * s/ sumP;
    % why not: H = exp(-sum(P(P > 1e-5) .* log(P(P > 1e-5)))); ???
    P = P / sumP;
    
end

