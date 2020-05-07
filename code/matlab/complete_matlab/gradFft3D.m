function [rep] = gradFft3D(Y,n,N_boxes_per_d,points_per_box)
    rep=zeros(n,3);
    charges=[Y ones(n,1)];
    phi=compute3D(Y,charges,N_boxes_per_d,points_per_box,1,n,4);
    charges=ones(n,1);
    phi2=compute3D(Y,charges,N_boxes_per_d,points_per_box,0,n,1);
    Z=sum(phi2(:));
    Z=Z-n;
    for i=1:n
        rep(i,1)=(-phi(i,1)+Y(i,1)*phi(i,4))/Z;
        rep(i,2)=(-phi(i,2)+Y(i,2)*phi(i,4))/Z;
        rep(i,3)=(-phi(i,3)+Y(i,3)*phi(i,4))/Z;
    end
    
    
    global error1
    global error2
    global speedup2
    
   
    precent=0.5;
    number=4;
    
    if N_boxes_per_d<30
        precent=0.8;
    end
    rep2=avgForces(number,precent,Y,N_boxes_per_d,1/(number+1),points_per_box);
    
    [~,Frep]=exactGradient(Y,0,n);
    error_1=norm(Frep-4*rep)/norm(Frep);
    error_2=norm(Frep-4*rep2)/norm(Frep);

    
    error1=[error1 error_1];
    error2=[error2 error_2];
    speedup2=[speedup2 speedup];
end

