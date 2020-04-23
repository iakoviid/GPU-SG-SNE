function [rep] =  compute3Dshift(Y,charges,Nint,k,squared,n,nsums,shift)


minY=min(Y(:))+shift;
minY=[minY minY minY];
maxY=max(Y(:))+shift;
maxY=[maxY maxY maxY];



box_width=(maxY(1)-minY(1))/Nint;
N1d = Nint * k;
total_boxes=Nint^3;
%Compute Box bounds
box_lower_bounds=zeros(total_boxes,3);
box_upper_bounds=zeros(total_boxes,3);
for i=1:Nint
    for j=1:Nint
        for z=1:Nint
            box_lower_bounds((i-1)*Nint^2+(j-1)*Nint+z,1)=box_width*(z-1)+minY(1);
            box_upper_bounds((i-1)*Nint^2+(j-1)*Nint+z,1)=box_width*(z)+minY(1);
            
            box_lower_bounds((i-1)*Nint^2+(j-1)*Nint+z,2)=box_width*(j-1)+minY(2);
            box_upper_bounds((i-1)*Nint^2+(j-1)*Nint+z,2)=box_width*(j)+minY(2);
            
            box_lower_bounds((i-1)*Nint^2+(j-1)*Nint+z,3)=box_width*(i-1)+minY(3);
            box_upper_bounds((i-1)*Nint^2+(j-1)*Nint+z,3)=box_width*(i)+minY(3);
        end
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
y_tilde=zeros(N1d,3);

y_tilde(1,:) = minY + h / 2;
for (i = 2:N1d)
    y_tilde(i,:)=y_tilde(i - 1,:) + h;
end


total_interp_point=N1d^3;



% We need to be able to look up which box each point belongs to
box_width=box_upper_bounds(1,1)-box_lower_bounds(1,1);
int_lookup = zeros(n,1);

for i=1:n
    current_intx = 0;
    current_inty=0;
    current_intz=0;

    current_intx=floor((Y(i,1)-box_lower_bounds(1,1))/box_width)+1;
    
    
    current_inty=floor((Y(i,2)-box_lower_bounds(1,1))/box_width)+1;
    current_intz=floor((Y(i,3)-box_lower_bounds(1,1))/box_width)+1;

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
    
    if (current_intz > Nint)
        current_intz = Nint;
    elseif (current_intz <= 0)
        current_intz = 1;
    end
    
    int_lookup(i) = current_intx+(current_inty-1)*Nint+(current_intz-1)*Nint^2;
end

%%  Compute the relative position of each point in its box in the interval [0, 1]
points_in_box = zeros(n,3);

for (i = 1: n)
    box_idx = int_lookup(i);
    x_min = box_lower_bounds(box_idx,1);
    y_min = box_lower_bounds(box_idx,2);
    z_min = box_lower_bounds(box_idx,3);

    points_in_box(i,:) = (Y(i,:)-[x_min y_min z_min])/box_width;
    
end


% Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients

%Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
Vx = zeros(n, k);
Vx=interpolate(k, n, points_in_box(:,1), interp_in_box);
%ompute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
Vy = zeros(n, k);
Vy=interpolate(k, n, points_in_box(:,2), interp_in_box);

Vz = zeros(n, k);
Vz=interpolate(k, n, points_in_box(:,3), interp_in_box);

w=zeros((Nint*k)^3,nsums);
for i=1:n
    box_idx=int_lookup(i)-1;
    box_z = floor(box_idx/Nint^2);
    box_i = mod(box_idx,Nint);
    box_j = box_idx -box_z*Nint^2-box_i;
    box_j=floor(box_j/Nint);
    for ( interp_i = 0:k-1)
        for ( interp_j  = 0:k-1)
            for ( interp_z  = 0:k-1)
            idx = ((box_i * k + interp_i) *(N1d) + (box_j * k + interp_j))*(N1d)+ (box_z * k) + interp_z;
            
            for nterms=1:nsums
                w(idx+1,nterms)= w(idx+1,nterms)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*Vz(i,interp_z+1)*charges(i,nterms);
            end
            
            end
            
        end
    end
end

%b=g2g3d(w,N1d,y_tilde,squared,nsums);
b1=g2g3dnopadd(w,N1d,y_tilde,squared,nsums);
fpol=zeros(n,nsums);

for i=1:n
    box_idx=int_lookup(i)-1;
    box_z = floor(box_idx/Nint^2);
    box_i = mod(box_idx,Nint);
    box_j = box_idx -box_z*Nint^2-box_i;
    box_j=floor(box_j/Nint);
    for ( interp_i = 0:k-1)
        for ( interp_j  = 0:k-1)
            for ( interp_z  = 0:k-1)
            idx = ((box_i * k + interp_i) *(N1d) + (box_j * k + interp_j))*(N1d)+ (box_z * k) + interp_z;
            
            for nterms=1:nsums
                fpol(i,nterms)= fpol(i,nterms)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*Vz(i,interp_z+1)*b1(mod(idx,N1d)+1,floor((idx-floor(idx/N1d^2)*N1d^2)/N1d)+1,floor(idx/N1d^2)+1,nterms);
%                 disp(b1(mod(idx,N1d)+1,floor((idx-floor(idx/N1d^2)*N1d^2)/N1d)+1,floor(idx/N1d^2)+1,nterms)-b(idx+1,nterms));
            end
            
            end
            
        end
    end
    
    
end
rep=fpol;

end

