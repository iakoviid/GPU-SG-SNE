function [weight] = t_dist(v,u)
    weight=1./(1+sum(v.^2)+sum(u.^2)-2*v*u');
end

