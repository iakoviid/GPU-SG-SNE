function [rep] =  compute2D(Y,charges,Nint,k,squared,n,nsums)
% locs=Y;
% flt2Dimproved;
minx=min(Y(:));
maxx=max(Y(:));
miny=minx;
maxy=maxx;

% minx=-50;
% maxx=50;
% miny=-50;
% maxy=50;

box_width=(maxx-minx)/Nint; %see it 
N1d = Nint * k;
total_boxes=Nint^2;
%Compute Box bounds
box_lower_bounds=zeros(2*total_boxes,1);
box_upper_bounds=zeros(2*total_boxes,1);
for i=1:Nint
    for j=1:Nint
        box_lower_bounds((i-1)*Nint+j)=box_width*(j-1)+minx;
        box_upper_bounds((i-1)*Nint+j)=box_width*j+minx;
        box_lower_bounds((i-1)*Nint+j+total_boxes)=box_width*(i-1)+miny;
        box_upper_bounds((i-1)*Nint+j+total_boxes)=box_width*(i)+miny;
    end
end

interp_in_box=zeros(k,1);
% Coordinates of each (equispaced) interpolation node for a single box
h = 1 / k;
interp_in_box(1) = h / 2;
for (i = 2:k) 
    interp_in_box(i) = interp_in_box(i - 1) + h;
end

%Coordinates of all the equispaced interpolation points
n_fft_coeffs = 2 * N1d;
h = h * box_width;
x_tilde=zeros(N1d,1);
y_tilde=zeros(N1d,1);

x_tilde(1) = minx + h / 2;
y_tilde(1) = miny + h / 2;
for (i = 2:N1d)
    x_tilde(i)=x_tilde(i - 1) + h;
    y_tilde(i)=y_tilde(i - 1) + h;
end

total_interp_point=N1d^2;



%% We need to be able to look up which box each point belongs to
box_width=box_upper_bounds(1)-box_lower_bounds(1);
int_lookup = zeros(n,1);

for i=1:n
    current_intx = 0;
    current_inty=0;
    
    current_intx=floor((Y(i,1)-box_lower_bounds(1))/box_width)+1;
    
    
    current_inty=floor((Y(i,2)-box_lower_bounds(1))/box_width)+1;
    if (current_intx > Nint)
        current_intx = Nint;
    elseif (current_intx <= 0)
        current_intx = 1;
    end
    if (current_inty > Nint)
        current_inty = Nint;
    elseif (current_inty <= 0)
        current_inty = 1;
    end
    
    int_lookup(i) = current_intx+(current_inty-1)*Nint;
end

%%  Compute the relative position of each point in its box in the interval [0, 1]
points_in_box = zeros(n,2);

for (i = 1: n)
    box_idx = int_lookup(i);
    x_min = box_lower_bounds(box_idx);
    y_min = box_lower_bounds(total_boxes + box_idx);
    points_in_box(i,:) = (Y(i,:)-[x_min y_min])/box_width;
    
end


%% Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients

%Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
Vx = zeros(n, k);
Vx=interpolate(k, n, points_in_box(:,1), interp_in_box);
%ompute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
Vy = zeros(n, k);
Vy=interpolate(k, n, points_in_box(:,2), interp_in_box);

w=zeros((Nint*k)^2,nsums);
for i=1:n
    box_idx=int_lookup(i)-1;
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
    for ( interp_i = 0:k-1)
        for ( interp_j  = 0:k-1)
            idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
            for nterms=1:nsums
                w(idx+1,nterms)= w(idx+1,nterms)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*charges(i,nterms);
            end
        end
    end
end
b=g2g2dnopadd2(w,N1d,x_tilde,y_tilde,squared,nsums);
fpol=zeros(n,nsums);

for i=1:n
    box_idx=int_lookup(i)-1;
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
    for ( interp_i = 0:k-1)
        for ( interp_j  = 0:k-1)
            idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
            for nterms=1:nsums
                 fpol(i,nterms)= fpol(i,nterms)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*b(floor(idx/N1d)+1,mod(idx,N1d)+1,nterms);
            end
        end
    end
end
rep=fpol;

end

