function [ val ] = bar_obj_val(x,y,ep,w_xi,t,C)
%calcualte the barrier objective value
    n = size(x,1);
    dplus1 = size(x,2);
    w = w_xi(1:dplus1);
    xi = w_xi(dplus1+1:end);
    dem1 = -y+x*w-ep*ones(n,1)-xi;
    dem2 = y-x*w-ep*ones(n,1)-xi;
    
    if any(-dem1<0) || any(-dem2<0) || any(xi<0)
        val = Inf;
        return;
    end
    
    obj_val = obj_val_orig(x,w_xi,C);
    
    term1 = -sum(log(-dem1));
    term2 = -sum(log(-dem2));
    term3 = -sum(log(xi));
    
    val = t*obj_val + term1 + term2 + term3;

end