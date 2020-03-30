function [result] = kernel(x,y,squared)
      result = sum((x-y).^2);
      result = 1/(1+result);
      if(squared)
        result=result^2;
      end
end

