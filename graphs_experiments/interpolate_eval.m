function [V] = interpolate_eval(locs,interp_points)
N=length(interp_points);
n=length(locs);
V = zeros(n,N); 
     for ti=1:N
          for yj=1:n
                num = 1;
                denom = 1;
                for tii=1:N
                    if (tii ~= ti)
                        denom = denom*(interp_points(ti) -interp_points(tii));
                        num= num*(locs(yj) - interp_points(tii));
                    end
                end

                V(yj,ti) = num/denom;
           end
     end
end

