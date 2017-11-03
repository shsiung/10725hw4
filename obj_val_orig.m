function [ obs_val ] = obj_val_orig(x,w_xi,C)
% Calculate the original objective value
    dplus1 = size(x,2);
    w = w_xi(1:dplus1-1);
    xi = w_xi(dplus1+1:end);

    obs_val = 0.5*(w'*w)+C*sum(xi);
end

