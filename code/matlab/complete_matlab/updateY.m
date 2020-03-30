function     [ydata,gains,y_incs]=updateY(ydata,y_incs,gains,min_gain,momentum,epsilon,y_grads)
    gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
        + (gains * .8) .* (sign(y_grads) == sign(y_incs));
    gains(gains < min_gain) = min_gain;
    y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
    ydata = ydata + y_incs;
    ydata = bsxfun(@minus, ydata, mean(ydata, 1));
  

end

