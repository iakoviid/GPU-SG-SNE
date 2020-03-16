function [ydata, ydataex] = tsne_compute(P,P_tilde, labels, no_dims,n)

if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    ydataex=no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Initialize some variables
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
max_iter = 10000;                                    % maximum number of iterations
epsilon = 500;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta

% Make sure P-vals are set properly
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
%const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
P = P * 4;                                      % lie about the P-vals to find better local minima

P_tilde(1:n + 1:end) = 0;                                 % set diagonal to zero
P_tilde = 0.5 * (P_tilde + P_tilde');                                 % symmetrize P-values
P_tilde = max(P_tilde ./ sum(P_tilde(:)), realmin);                   % make sure P-values sum to one
P_tilde = P_tilde * 4;                                      % lie about the P-vals to find better local minima


% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end

y_incs  = zeros(size(ydata));
gains = ones(size(ydata));

y_incs2  = zeros(size(ydata));
gains2 = ones(size(ydata));

% Run the iterations
for iter=1:max_iter
    [Fatr,Frep]=exactGradient(ydata,P,n);
    [Fatr_tilde,Frep_tilde]=aproxGradient(ydata,P_tilde,no_dims,n);

    y_grads=Fatr-Frep;
    grad_aprox=Fatr_tilde-Frep_tilde;
    
    error_rep(iter)=norm(Frep-Frep_tilde)/norm(Frep);
    error_rep(iter)=log10(error_rep(iter));
    
    error_atr(iter)=norm(Fatr-Fatr_tilde)/norm(Fatr);
    error_atr(iter)=log10(error_atr(iter));
    
    error_grad(iter)=norm(y_grads-grad_aprox)/norm(y_grads);
    error_grad(iter)=log10(error_grad(iter));
    % Update the solution
    [ydata,gains,y_incs]=updateY(ydata,y_incs,gains,min_gain,momentum,epsilon,grad_aprox);
    
    [Fatr,Frep]=exactGradient(ydataex,P,n);
    y_grads=Fatr-Frep;
    [ydataex,gains2,y_incs2]=updateY(ydataex,y_incs2,gains2,min_gain,momentum,epsilon,y_grads);
    
    % Update the momentum if necessary
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter
        P = P ./ 4;
        P_tilde=P_tilde./4;
    end
    
end
figure();
plot(error_rep,'linewidth',4)
xlabel('Iteration')
ylabel('log10(RRSE)')
title('Repulsive Error for 1000 Gradient Descent Iterations')
figure();
plot(error_atr,'linewidth',4)
xlabel('Iteration')
ylabel('log10(RRSE)')
title('Attractive Error for 1000 Gradient Descent Iterations')

figure();
plot(error_grad,'linewidth',4)
xlabel('Iteration')
ylabel('log10(RRSE)')
title('Total Gradient Error for 1000 Gradient Descent Iterations')

end
