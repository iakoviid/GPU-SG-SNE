function [outputArg1] = displaysouts(Y,f,phi,phistar,number,time1,time2,k,N2,width,scale)
fprintf("Error phi=%f \n",norm(f-phi)/norm(f));
fprintf("Error phistar=%f \n",norm(f-phistar)/norm(f));
fprintf("ration = %f\n",norm(f-phi)/norm(f-phistar));
miny=min(Y(:));
h=width/(N2*k);
N1d=N2*k;
figure();
labs=[];
for i=0:number
    y_tilde=zeros(N1d,1);
    y_tilde(1) = miny+h/2+i*scale;
    for (j = 2:N1d)
        y_tilde(j)=y_tilde(j - 1) + h;
    end
    plot(y_tilde,zeros(size(y_tilde)),"o");
    labs=[labs "p="+string(i)];
    hold on;
end
plot(min(Y(:))*ones(size([min(Y(:)):1:max(Y(:))])),[min(Y(:)):1:max(Y(:))]);
plot(max(Y(:))*ones(size([min(Y(:)):1:max(Y(:))])),[min(Y(:)):1:max(Y(:))]);
labs=[labs "max" "min"];
hold off
errorRation=norm(f-phi)/norm(f-phistar);
speedup=time1/time2;
fprintf("speedup=%f\n",speedup);
legend(labs);
end

