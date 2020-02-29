clear all;
rng(1)
a = 0; b=1; n = 1000;
locs = rand(n,2)*(b-a);
%n=2;
%locs=[0.3 0.3;0.1 0.7];

distmatrix = squareform(pdist(locs));
kernel = 1./(1+distmatrix.^2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here

f = kernel*v;%Result
%% Number of Nint intervals and k interpolation points per dimension make Nint^2 boxes with k^2 points inside them interval h interval
%length
error=zeros(20,1);
for (papa=2:20)
k = papa;
Nint = papa;
%h = 1/(Nint *k);
total_boxes=Nint^2;
box_width=1/Nint;
box_lower_bounds=zeros(2*total_boxes,1);
box_upper_bounds=zeros(2*total_boxes,1);

for i=1:Nint
    for j=1:Nint
        box_lower_bounds((i-1)*Nint+j)=0+box_width*(j-1);
        box_upper_bounds((i-1)*Nint+j)=0+box_width*j;
        box_lower_bounds((i-1)*Nint+j+total_boxes)=0+box_width*(i-1);
        box_upper_bounds((i-1)*Nint+j+total_boxes)=0+box_width*(i);
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
N1d = Nint * k;
n_fft_coeffs = 2 * N1d;
h = h * box_width;
x_tilde=zeros(N1d,1);
y_tilde=zeros(N1d,1);

x_tilde(1) = 0 + h / 2;
y_tilde(1) = 0 + h / 2;
for (i = 2:N1d) 
        x_tilde(i) = x_tilde(i - 1) + h;
        y_tilde(i) = y_tilde(i - 1) + h;
end


kernel_tilde=zeros(2*N1d,2*N1d);
for i = 0:N1d-1
    for j =0:N1d-1
        tmp = sqrt((x_tilde(1) -x_tilde(i+1))^2 +(x_tilde(1) -x_tilde(j+1) )^2);
        tmp = 1/(1+tmp^2);
        kernel_tilde((N1d + i)+1 , (N1d + j)+1) = tmp;
        kernel_tilde((N1d - i)+1 , (N1d + j)+1) = tmp;
        kernel_tilde((N1d + i)+1 , (N1d - j)+1) = tmp;
        kernel_tilde((N1d - i)+1 , (N1d - j)+1) = tmp;
    end
end

fft_kernel=fft2(kernel_tilde);

total_interp_point=N1d^2;


%% We need to be able to look up which box each point belongs to
int_lookup = zeros(n,1);

for i=1:n
    current_intx = 0;
    current_inty=0;
    while (k*h*(current_intx) < locs(i,1))
        current_intx = current_intx +1;
    end
    if(current_intx==floor((locs(i,1)-box_lower_bounds(1))/box_width)+1)
     else
                disp("No");

    end
    while (k*h*(current_inty) < locs(i,2))
        current_inty = current_inty +1;
    end
    
    if(current_inty==floor((locs(i,2)-box_lower_bounds(1))/box_width)+1)
    else
                disp("No");

    end
   
   int_lookup(i) = current_intx+(current_inty-1)*Nint;
end
%%  Compute the relative position of each point in its box in the interval [0, 1]
    points_in_box = zeros(n,2);
    
    for (i = 1: n) 
         box_idx = int_lookup(i);
         x_min = box_lower_bounds(box_idx);
         y_min = box_lower_bounds(total_boxes + box_idx);
        points_in_box(i,:) = (locs(i,:)-[x_min y_min])/box_width;
      
    end

    
  %% Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
    
  %Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
  Vx = zeros(n, k);
  Vx=interpolate(k, n, points_in_box(:,1), interp_in_box);
  %ompute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
  Vy = zeros(n, k);
  Vy=interpolate(k, n, points_in_box(:,2), interp_in_box);

w=zeros((Nint*k)^2,2);
for i=1:n
    box_idx=int_lookup(i)-1;
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
      for ( interp_i = 0:k-1) 
            for ( interp_j  = 0:k-1)  
                 idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
                    w(idx+1,1)= w(idx+1,1)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*v(i,1);
                    w(idx+1,2)=w(idx+1,2)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*v(i,2);

            end
      end
end
b=zeros(N1d^2,2);

fa=vec2mat(w(:,1),N1d);
fa=[fa zeros(N1d,N1d);zeros(N1d,2*N1d)];

result=ifft2(fft2(kernel_tilde).*fft2(fa));
result= (result(N1d+1:2*N1d,N1d+1:2*N1d))';
b(:,1)=result(:);
fa=vec2mat(w(:,2),N1d);
fa=[fa zeros(N1d,N1d);zeros(N1d,2*N1d)];
result=ifft2(fft2(kernel_tilde).*fft2(fa));
result= (result(N1d+1:2*N1d,N1d+1:2*N1d))';
b(:,2)=result(:);

fpol=zeros(n,2);
for i=1:n
    box_idx=int_lookup(i)-1;
    box_j = floor(box_idx/Nint);
    box_i = mod(box_idx,Nint);
      for ( interp_i = 0:k-1) 
            for ( interp_j  = 0:k-1)  
                 idx = (box_i * k + interp_i) * (Nint * k) + (box_j * k) + interp_j;
                    fpol(i,1)= fpol(i,1)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*b(idx+1,1);
                    fpol(i,2)=fpol(i,2)+Vx(i,interp_i+1)*Vy(i,interp_j+1)*b(idx+1,2);

            end
      end
end
error(papa)=norm(f-fpol)/norm(f);
error(papa)=log10(error(papa));
fprintf("Nint=k=%d error=%f\n",k,error(papa));
end
plot(error);
title('log10(Relative Error) 2D with box interp ');
xlabel('Nboxes/dim=points/box');