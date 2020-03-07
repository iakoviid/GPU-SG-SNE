function [rep] =  compute1D(Y,charges,Nint,k,squared,n,nsums)

minY=min(Y);
maxY=max(Y);

box_width=(maxY-minY)/Nint;
N1d = Nint * k;
%Compute Box bounds
box_lower_bounds=zeros(Nint,1);
box_upper_bounds=zeros(Nint,1);
for i=1:Nint
        box_lower_bounds(i)=box_width*(i-1)+minY;
        box_upper_bounds(i)=box_width*i+minY;
    
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
x_tilde(1) = minY + h / 2;
for (i = 2:N1d)
    x_tilde(i)=x_tilde(i - 1) + h;
end


kernel_tilde=zeros(2*N1d,1);
for i = 0:N1d-1

        tmp=kernel(x_tilde(1),x_tilde(i+1),squared);

        for signi=-1:2:1
                kernel_tilde((N1d +signi*i)+1 ) = tmp;
            
        end
    
end

fft_kernel=fft(kernel_tilde);




% We need to be able to look up which box each point belongs to
box_width=box_upper_bounds(1)-box_lower_bounds(1);
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

%%  Compute the relative position of each point in its box in the interval [0, 1]
points_in_box = zeros(n,1);

for (i = 1: n)
    box_idx = int_lookup(i);
    x_min = box_lower_bounds(box_idx);
    points_in_box(i) = (Y(i)-x_min)/box_width;
    
end


%% Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients

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

b=zeros(N1d,nsums);
for nterms=1:nsums
    fa=w(:,nterms);
    fa=[zeros(N1d,1); fa ];
    result=ifft(fft_kernel.*fft(fa));
    b(:,nterms)= result(1:N1d);

end


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

