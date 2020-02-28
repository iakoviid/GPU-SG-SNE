clear all;
rng(1)
a = 0; b=1; n = 2;
%[locs,~] = sort(rand(n,2)*(b-a));
locs=[0.3 0.3;0.1 0.7];
kernel = kerneleval1(locs);
v=zeros(n,2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here
f=mult2(kernel,v);

N = 2; 
h = 1/(N);
interp_points=zeros(N,2);
for(j=1:N)
        interp_points(j,1)=h*(j-1)+h/2;
        interp_points(j,2)=h*(j-1)+h/2;
        
end

Vx=interpolate_eval(locs(:,1),interp_points(:,1));
Vy=interpolate_eval(locs(:,2),interp_points(:,2));
k=N^2;
V=zeros(k,n);
for i=1:k
    for j=1:n
        V(i,j)=Vx(j,mod(i,N)+1)*Vy(j,ceil(i/N));
    end
end

a=mult2d(V,v);



