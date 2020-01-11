%%
clear all;
rng(1)
a = 0; b=1; n = 5;
[locs,~] = sort(rand(n,2)*(b-a));
distmatrix = squareform(pdist(locs));
kernel = 1./(1+distmatrix.^2);
v=zeros(n,2);
v(:,1) = sin(10*locs(:,1)) + cos(2000*locs(:,1)); %Anything could be used here
v(:,2) = sin(10*locs(:,2)) + cos(2000*locs(:,2)); %Anything could be used here
f=zeros(n,2);
for(i=1:n)
    for(j=1:n)
        f(i,:)=f(i,:)+kernel(i,j)*v(j,:);
    end
end

%% Number of Nint intervals k interpolation points per interval h interval
%length
N = 2; 
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
    
 Vx = zeros(n,N); %Columns of V will form our polynomial basis

 for ti=1:N
      for yj=1:n
            num = 1;
            denom = 1;
            for tii=1:N
                if (tii ~= ti)
                    denom = denom*(interp_points(ti,2) -interp_points(tii,2));
                    num= num*(locs(yj,1) - interp_points(tii,2));
                end
            end

            Vx(yj,ti) = num/denom;
       end
 end
 Vy = zeros(n,N); %Columns of V will form our polynomial basis

 for ti=1:N
      for yj=1:n
            num = 1;
            denom = 1;
            for tii=1:N
                if (tii ~= ti)
                    denom = denom*(interp_points(ti,2) -interp_points(tii,2));
                    num= num*(locs(yj,2) - interp_points(tii,2));
                end
            end

            Vy(yj,ti) = num/denom;
       end
 end
V=zeros(k,n);
for i=1:k
    for j=1:n
        V(i,j)=Vx(j,mod(i,N)+1)*Vy(j,ceil(i/N));
    end
end

S=zeros(k,k);
for(i=1:k)
    for(j=1:k)
        S(i,j)=1/(norm(interp_points(i,:)-interp_points(j,:))+1)^2;
    end
end


Vt=V';
a=zeros(k,2);
for(i=1:k)
    for(j=1:n)
        a(i,:)=a(i,:)+V(i,j)*v(j,:);
    end
end




b=zeros(k,2);
for(i=1:k)
    for(j=1:k)
        b(i,:)=b(i,:)+S(i,j)*a(j,:);
    end
end


fpol=zeros(n,2);
for(i=1:n)
    for(j=1:k)
        fpol(i,:)=fpol(i,:)+V(j,i)*b(j,:);
    end
end

norm(f(:,1)-fpol(:,1))+norm(f(:,2)-fpol(:,2));



