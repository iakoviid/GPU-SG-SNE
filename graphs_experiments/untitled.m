rng(1)
a = 0; b=1; n = 1000;
[locs,~] = sort(rand(n,1)*(b-a));
distmatrix = squareform(pdist(locs));
kernel1 = 1./(1+distmatrix.^2); 
kernel2 = 1./(distmatrix.^2);kernel2(kernel2==Inf) = 0;


[U1, S1, V1] = svd(kernel1); 
[U2, S2, V2] = svd(kernel2); 
figure(1)
semilogy(diag(S1), 'linewidth',4); hold on
semilogy(diag(S2), 'linewidth',4);
legend('K_1', 'K_2'); title('Spectra of Kernels: Neear Field Interactions'); set(gca,'FontSize',12)

q = sin(10*locs) + cos(2000*locs); %Anything could be used here

%Exact 
f1 = kernel1*q;
f2 = kernel2*q;

ks = [1:20 100 200];
K1_svd_errors = ones(length(ks),1); K2_svd_errors = ones(length(ks),1);
for ki =1:length(ks),
    k = ks(ki);
    kernel1_approx = V1(:,1:k) * S1(1:k,1:k) * V1(:,1:k)';
    f1_approx = kernel1_approx*q;
    K1_svd_errors(ki) = norm(f1_approx - f1)/norm(f1);
    
    kernel2_approx = V2(:,1:k) * S2(1:k,1:k) * V2(:,1:k)';
    f2_approx = kernel2_approx*q;
    K2_svd_errors(ki) = norm(f2_approx - f2)/norm(f2);
end

figure(2); clf
semilogy(ks,K1_svd_errors,'linewidth',2);
hold on
semilogy(ks,K2_svd_errors,'linewidth',2);
ylabel ('Relative error'); xlabel('k'); set(gca,'FontSize',12); title('Optimal Rank-k Approximation: Near Field');
legend('K_1', 'K_2'); 

a1 = 0; b1=1; 
a2 = 2; b2 = 3;
n = 1000;
[locs1,~] = sort((b1+a1)/2+ rand(n,1)*(b1-a1));
[locs2,~] = sort((b2+a2)/2+rand(n,1)*(b2-a2));

distmatrix = squareform(pdist([locs1; locs2]));
kernel1 = 1./(1+distmatrix(1:1000,1001:2000).^2); 
kernel2 = 1./(distmatrix(1:1000,1001:2000).^2);kernel2(kernel2==Inf) = 0;

[U1, S1, V1] = svd(kernel1); 
[U2, S2, V2] = svd(kernel2); 
figure(100)
semilogy(diag(S1), 'linewidth',4); %xlim([0,500]);
 hold on
semilogy(diag(S2), 'linewidth',4);%xlim([0,500]);
legend('K_1', 'K_2'); title('Spectrum of kernels'); set(gca,'FontSize',12)

ps = 1:15;
K1_poly_errors = ones(length(ps),1);

%With varying number of interpolation points
for pi=1:length(ps)
    h = (b-a)/ps(pi); % Distance between interpolation points
    interp_points = a:h:b;
    k = length(interp_points);% Number of interpolation points
    
    V = zeros(n,k); %Columns of V will form our polynomial basis

    % There are k Lagrange polynomials (one for each point), evaluate each
    % of them at all the n points
    
    %Note how this is entirely independent of the kernel!
    for ti=1:length(interp_points),
        for yj=1:n
            num = 1;
            denom = 1;
            for tii=1:k
                if (tii ~= ti)
                    denom = denom*(interp_points(ti) -interp_points(tii));
                    num= num*(locs(yj) - interp_points(tii));
                end
            end

            V(yj,ti) = num/denom;
        end
    end

    %We only evaluate the kernel at the k by k interpolation points
    S = ones(k,k);
    for i=1:k
        for j=1:k       
            S(i,j) = 1/(1+norm(interp_points(j)-interp_points(i))^2);
        end
    end
    
    f1_poly_approx = V*S*V'*q;
    K1_poly_errors(pi) = norm(f1_poly_approx - f1)/norm(f1);
end

figure(5); clf
semilogy(ps,K1_poly_errors,'linewidth',2);
hold on
semilogy(ks(1:20),K1_svd_errors(1:20), 'linewidth',2)
ylabel ('Relative error'); xlabel('k'); set(gca,'FontSize',12); title('Polynomial rank-k approximation');
legend('Lagrange Polynomial', 'SVD')

