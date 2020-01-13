function [Au] = BT_fft(n,f,n_ind,level)
if nargin <3
    n_ind=ones(2,length(n));
    level=1;
end
if level==(length(n)-1)
    Au=flipud(application_function(f));
    Au=Au(:);
else
    this_n=n(level);
    for i=this_n:-1:1
        b_edge=f^2*prod(2*n(level(n))-1);
        n_ind(1,level)=i;
	blk=BT_fft(n,f,n_ind,level+1);
	Au(1+b_edge*(this_n-i):b_edge*(this_n-i+1))=blk;
end

        
end

