function [L] = interpolate(k, n, points_in_box, interp_in_box)
L = zeros(n, k);
denominator = zeros(k,1);
for (i= 1: k)
    denominator(i) = 1;
    for (j = 1: k)
        if (i ~= j)
            denominator(i) = denominator(i)*(interp_in_box(i) - interp_in_box(j));
        end
    end
end
for (i = 1:n) 
    for ( j = 1:k) 
        L(i,j) = 1;
        for ( m = 1:k) 
            if (j ~= m) 
                L(i,j) = L(i,j)*(points_in_box(i) - interp_in_box(m));
            end
        end
        L(i,j) = L(i,j)/denominator(j);
    end
end



end

