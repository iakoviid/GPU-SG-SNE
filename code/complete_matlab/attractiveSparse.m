function [atr] = attractiveSparse(ydata,P)
    [i,j,v]=find(P);
    for t=1:length(i)
        weight=t_dist(ydata(i(t),:),ydata(j(t),:));
        v(t)=v(t)*weight;
    end
        
    PP=sparse(i,j,v);
    atr=4*(diag(sum(PP, 1)) - PP) * ydata;

end

