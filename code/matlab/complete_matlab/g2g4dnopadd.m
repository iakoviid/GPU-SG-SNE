function [b] = g2g4dnopadd(w,N1d,y_tilde,squared,nsums)
v=zeros(N1d,N1d,N1d,N1d,nsums);
b=zeros(N1d,N1d,N1d,N1d,nsums);
wc = exp( -2*pi*1i*[0:N1d-1]'/(2*N1d) );
conjwc=conj(wc);
for nterms=1:nsums
    for(i=1:N1d)
        for(j=1:N1d)
            for(z=1:N1d)
                for(t=1:N1d)
                    
                    v(i,j,z,t,nterms)=w((j-1)*N1d+i+(z-1)*N1d^2+(t-1)*N1d^3,nterms);
                end
            end
        end
    end
end
for signt=1:-2:-1
    for signz=1:-2:-1
        for signj=1:-2:-1
            for signi=1:-2:-1
                Kc=zeros(N1d,N1d,N1d,N1d);
                for nterms=1:nsums
                    for(i=1:N1d)
                        for(j=1:N1d)
                            for(z=1:N1d)
                                for(t=1:N1d)
                                    
                                    v(i,j,z,t,nterms)=w((j-1)*N1d+i+(z-1)*N1d^2+(t-1)*N1d^3,nterms);
                                end
                            end
                        end
                    end
                end
                for i = 0:N1d-1
                    for j =0:N1d-1
                        for z=0:N1d-1
                            for t=0:N1d-1
                                tmp=kernel([y_tilde(1,1) y_tilde(1,2) y_tilde(1,3) y_tilde(1,4)],[y_tilde(i+1,1) y_tilde(j+1,2) y_tilde(z+1,3) y_tilde(t+1,4)],squared);
                                Kc(i+1,j+1,z+1,t+1) =Kc(i+1,j+1,z+1,t+1)+ tmp;
                                if (i>0) Kc(N1d-i+1,j+1,z+1,t+1) =Kc(N1d-i+1,j+1,z+1,t+1)+signi*tmp; end
                                if (j>0) Kc(i+1,N1d-j+1,z+1,t+1) =Kc(i+1,N1d-j+1,z+1,t+1)+ signj*tmp;end
                                if (i>0 && j>0) Kc(N1d-i+1,N1d-j+1,z+1,t+1) =Kc(N1d-i+1,N1d-j+1,z+1,t+1)+ signj*signi*tmp ;end
                                if (z>0) Kc(i+1,j+1,N1d-z+1,t+1)=Kc(i+1,j+1,N1d-z+1,t+1)+ signz*tmp; end
                                if (z>0 && i>0) Kc(N1d-i+1,j+1,N1d-z+1,t+1) =Kc(N1d-i+1,j+1,N1d-z+1,t+1)+ signi*signz*tmp;end
                                if (z>0 && j>0) Kc(i+1,N1d-j+1,N1d-z+1,t+1) =Kc(i+1,N1d-j+1,N1d-z+1,t+1)+ signj*signz*tmp;end
                                if (z>0 && i>0 && j>0) Kc(N1d-i+1,N1d-j+1,N1d-z+1,t+1)=Kc(N1d-i+1,N1d-j+1,N1d-z+1,t+1)+ signi*signz*signj*tmp;end
                                if (t>0) Kc(i+1,j+1,z+1,N1d-t+1)=Kc(i+1,j+1,z+1,N1d-t+1)+ signt*tmp; end
                                if (z>0 && t>0) Kc(i+1,j+1,N1d-z+1,N1d-t+1) =Kc(i+1,j+1,N1d-z+1,N1d-t+1)+ signt*signz*tmp;end
                                if (t>0 && i>0) Kc(N1d-i+1,j+1,z+1,N1d-t+1) =Kc(N1d-i+1,j+1,z+1,N1d-t+1)+ signi*signt*tmp;end
                                if (t>0 && j>0) Kc(i+1,N1d-j+1,z+1,N1d-t+1) =Kc(i+1,N1d-j+1,z+1,N1d-t+1)+ signj*signt*tmp;end
                                if (t>0 && i>0 && j>0) Kc(N1d-i+1,N1d-j+1,z+1,N1d-t+1)=Kc(N1d-i+1,N1d-j+1,z+1,N1d-t+1)+ signi*signt*signj*tmp;end
                                if (t>0 && z>0 && i>0) Kc(N1d-i+1,j+1,N1d-z+1,N1d-t+1) =Kc(N1d-i+1,j+1,N1d-z+1,N1d-t+1)+ signt*signi*signz*tmp;end
                                if (t>0 && z>0 && j>0) Kc(i+1,N1d-j+1,N1d-z+1,N1d-t+1) =Kc(i+1,N1d-j+1,N1d-z+1,N1d-t+1)+ signt*signj*signz*tmp;end
                                if (t>0 && z>0 && i>0 && j>0) Kc(N1d-i+1,N1d-j+1,N1d-z+1,N1d-t+1)=Kc(N1d-i+1,N1d-j+1,N1d-z+1,N1d-t+1)+ signt*signi*signz*signj*tmp;end
                                
                                
                                
                            end
                        end
                    end
                end
                
                if(signt==-1)
                    for indexz=1:N1d
                        Kc(:,:,:,indexz)=Kc(:,:,:,indexz)*wc(indexz);
                        v(:,:,:,indexz,:)=v(:,:,:,indexz,:)*wc(indexz);
                    end
                end
                
                if(signz==-1)
                    for indexz=1:N1d
                        Kc(:,:,indexz,:)=Kc(:,:,indexz,:)*wc(indexz);
                        v(:,:,indexz,:,:)=v(:,:,indexz,:,:)*wc(indexz);
                    end
                end
                if(signi==-1)
                    
                    for(indexi=1:N1d)
                        Kc(indexi,:,:,:)=wc(indexi)*Kc(indexi,:,:,:);
                        for nterms=1:nsums
                            v(indexi,:,:,:,nterms)=wc(indexi)*v(indexi,:,:,:,nterms);
                        end
                    end
                end
                
                if(signj==-1)
                    for(indexi=1:N1d)
                        Kc(:,indexi,:,:)=wc(indexi)*Kc(:,indexi,:,:);
                        for nterms=1:nsums
                            v(:,indexi,:,:,nterms)=wc(indexi)*v(:,indexi,:,:,nterms);
                        end
                    end
                end
                
                for nterms=1:nsums
                    result(:,:,:,:,nterms)=ifftn(fftn(v(:,:,:,:,nterms)).*fftn(Kc));
                end
                
                if(signt==-1)
                    for indexz=1:N1d
                        result(:,:,:,indexz,:)=result(:,:,:,indexz,:)*conj(wc(indexz));
                    end
                end
                if(signz==-1)
                    for indexz=1:N1d
                        result(:,:,indexz,:,:)=result(:,:,indexz,:,:)*conj(wc(indexz));
                    end
                end
                if(signi==-1)
                    for(indexi=1:N1d)
                        for nterms=1:nsums
                            result(indexi,:,:,:,nterms)=conjwc(indexi)*result(indexi,:,:,:,nterms);
                        end
                    end
                end
                
                if(signj==-1)
                    for(indexi=1:N1d)
                        for nterms=1:nsums
                            result(:,indexi,:,:,nterms)=conjwc(indexi)*result(:,indexi,:,:,nterms);
                        end
                    end
                end
                b=b+real(result);
            end
        end
    end
end

b=b/16;
end

