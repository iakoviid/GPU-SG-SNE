%%
clear all;
rng(1)
a = 0; b=1; n = 2;
%[locs,~] = sort(rand(n,2)*(b-a));
locs=[0.3 0.3;0.1 0.7];
kernel = kerneleval1(locs);
v=zeros(n,2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here
f=mult2d(kernel,v);

%% Number of Nint intervals k interpolation points per interval h interval
%length
N = 3; 
h = 1/(N);
interp_points=zeros(N^2,2);
for(i=1:N)
    for(j=1:N)
        interp_points((i-1)*N+j,1)=h*(i-1)+h/2;
        interp_points((i-1)*N+j,2)=h*(j-1)+h/2;
        
    end
end

%% Lagrange Polynomials
 k = length(interp_points);% Number of interpolation points
 Vx=interpolate_eval(locs(:,1),interp_points(1:N,2));
 Vy=interpolate_eval(locs(:,2),interp_points(1:N,2));
V=zeros(k,n);
for i=1:k
    for j=1:n
        V(i,j)=Vx(j,mod(i,N)+1)*Vy(j,ceil(i/N));
    end
end

S = kerneleval1(interp_points);


Vt=V';
a=mult2d(V,v);
b=mult2d(S,a);
fpol=mult2d(V',b);

error=norm(f(:,1)-fpol(:,1))+norm(f(:,2)-fpol(:,2));
