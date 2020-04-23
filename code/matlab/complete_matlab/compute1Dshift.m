function [rep] =  compute1Dshift(Y,charges,Nint,k,squared,n,nsums,shift)

minY=min(Y)+shift;
maxY=max(Y)+shift;

box_width=(maxY-minY)/Nint;
N1d = Nint * k;

%Compute Box bounds
box_lower_bounds=zeros(Nint,1);
for i=1:Nint
        box_lower_bounds(i)=box_width*(i-1)+minY;
end

h = 1 / k;

interp_in_box=h/2:h:k*h-h/2;

h = h * box_width;

%creatLookUp1D()
% We need to be able to look up which box each point belongs to
int_lookup = zeros(n,1);

for i=1:n
    
    current_intx=floor((Y(i,1)-box_lower_bounds(1))/box_width)+1;
    if (current_intx > Nint)
        current_intx = Nint;
    elseif (current_intx <= 0)
        current_intx = 1;
    end
    int_lookup(i) = current_intx;
end

%  Compute the relative position of each point in its box in the interval [0, 1]
points_in_box = zeros(n,1);

for (i = 1: n)
    box_idx = int_lookup(i);
    x_min = box_lower_bounds(box_idx);
    points_in_box(i) = (Y(i)-x_min)/box_width;
    
end



%Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
Vx = zeros(n, k);
Vx=interpolate(k, n, points_in_box, interp_in_box);

w=zeros(Nint*k,nsums);
for i=1:n
    box_idx=(int_lookup(i)-1)*k;
    for ( interp_i = 0:k-1)
            idx = box_idx + interp_i;
            for nterms=1:nsums
                w(idx+1,nterms)= w(idx+1,nterms)+Vx(i,interp_i+1)*charges(i,nterms);
            end
    end
end

[b] = g2g1dnopadd(w,N1d,box_width/k,squared,nsums);

fpol=zeros(n,nsums);

for i=1:n
    box_idx=(int_lookup(i)-1)*k;
    for ( interp_i = 0:k-1)
            idx = box_idx + interp_i;
            for nterms=1:nsums
                 fpol(i,nterms)= fpol(i,nterms)+Vx(i,interp_i+1)*b(idx+1,nterms);
            end
        
    end
end
rep=fpol;

end

