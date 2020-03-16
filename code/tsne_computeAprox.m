function [ydata] = tsne_computeAprox(P_tilde, labels, no_dims,n)

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


P_tilde(1:n + 1:end) = 0;                                 % set diagonal to zero
%P_tilde = 0.5 * (P_tilde + P_tilde');                                 % symmetrize P-values
P_tilde = (P_tilde ./ sum(P_tilde(:)));                   % make sure P-values sum to one
P_tilde = P_tilde * 4;                                      % lie about the P-vals to find better local minima


% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end

y_incs  = zeros(size(ydata));
gains = ones(size(ydata));


% Run the iterations
for iter=1:max_iter
    [Fatr_tilde,Frep_tilde]=aproxGradient(ydata,P_tilde,no_dims,n);

    grad_aprox=Fatr_tilde-Frep_tilde;
    

    % Update the solution
    [ydata,gains,y_incs]=updateY(ydata,y_incs,gains,min_gain,momentum,epsilon,grad_aprox);

    % Update the momentum if necessary
    if iter == mom_switch_iter
        momentum = final_momentum;
    end
    if iter == stop_lying_iter
        P_tilde=P_tilde./4;
    end
    
end
    figure();
    gscatter(ydata(:,1),ydata(:,2),labels);
    title("MNIST "+n+" sampled points approx t-SNE");

end
