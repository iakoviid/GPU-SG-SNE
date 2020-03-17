function [X] = preproccessing(train_X)

    % Normalize input data
    train_X = train_X - min(train_X(:));
    train_X = train_X / max(train_X(:));
    train_X = bsxfun(@minus, train_X, mean(train_X, 1));

    X=train_X;
    % Perform preprocessing using PCA
    disp('Preprocessing data using PCA...');
    if size(X, 2) < size(X, 1)
        C = X' * X;
    else
        C = (1 / size(X, 1)) * (X * X');
    end
    [M, lambda] = eig(C);
    [lambda, ind] = sort(diag(lambda), 'descend');
    M = M(:,ind(1:initial_dims));
    lambda = lambda(1:initial_dims);
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');
    end
    X = bsxfun(@minus, X, mean(X, 1)) * M;
    clear M lambda ind

end

