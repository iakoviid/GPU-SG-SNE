%%
clear all
error=zeros(1,15);
for N=2:15
rng(1)
a = 0; b=1; n = 5000;
[locs,~] = sort(rand(n,2)*(b-a));
%locs=[0.3 0.3;0.1 0.7];
kernel = kerneleval1(locs);
v=zeros(n,2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here
f=kernel*v;

%% Number of Nint intervals k interpolation points per interval h interval
%length
 
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
for i=0:N-1
    for j=1:N
        V(i*N+j,:)=Vx(i+1,:).*Vy(j,:);
    end
end


%%  

S = kerneleval1(interp_points);

a=V*v;
b=S*a;
fpol=V'*b;

%% 
error(N)=norm(f(:,1)-fpol(:,1))/norm(f(:,1))+norm(f(:,2)-fpol(:,2))/norm(f(:,2));
end
plot(error)