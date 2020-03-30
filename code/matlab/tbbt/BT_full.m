 function  [A]=BT_full(n,f,n_ind,level)
 % ABT  full n,f,n  inf,level
%Recursive  multilevel  block  Toeplitz   MBT   matrix generator.
%    nn1 n2...nM  is the number  of BT blocks  ateach level.   
%f is the size of the final dense fxf block.
%n_ind  is  size  2xn  and  indicates  the  current  blockindex.
%Row 1down, Row 2across, 
%level indicates which level is current. 
%BT  full  is  initially  called  with  only  n  and  f  asarguments.
 if nargin<3
     n_ind=ones(2,length(n));
     level=1;
 end
 if level==length(n)+1
     A=application_function(n,f,n_ind,level);   %fxf   blockassignment.
 else
     this_n=n(level) ;
     for  i=1:this_n  %Lower  triangle  and  diagonal  assign-ment.
         b_edge=prod(n(level+1:length(n)))*f;
         n_ind(1,level)=i;
         blk=BT_full(n,f,n_ind,level+1);
         for j=1:(this_n-i+1)
             A(b_edge*(i-1)+[b_edge*(j-1)+1:b_edge*j],[b_edge*(j-1)+1:b_edge*j])=blk;
         end
     end
     for i=2:this_n %Upper triangle assignment.
         n_ind(2,level)=i;
         blk=BT_full(n,f,n_ind,level+1);
         for j=1:this_n-i+1
             A([b_edge*(j-1)+1:b_edge*j],b_edge*(i-1)+[b_edge*(j-1)+1:b_edge*j])=blk;
         end
     end
 end