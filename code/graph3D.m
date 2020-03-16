function [] = graph3D(ydata,labels)
figure();
for i=1:10
    idx=labels==i;
    scatter3(ydata(idx,1),ydata(idx,2),ydata(idx,3));
    hold on;
end
end

