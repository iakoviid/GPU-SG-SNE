function [outputArg1,outputArg2] = creatLookUp1D(inputArg1,inputArg2)

for i=1:n
    
    current_intx=floor((Y(i,1)-box_lower_bounds(1))/box_width)+1;
    
    
    if (current_intx > Nint)
        current_intx = Nint;
    elseif (current_intx <= 0)
        current_intx = 1;
    end
    int_lookup(i) = current_intx;
end

end

