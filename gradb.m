function [ gradf ] = gradb(x,y,ep,w_xi,c,t)
%find the gradient of f
    n = size(x,1);
    dplus1 = size(x,2);
    w = w_xi(1:dplus1);
    xi = w_xi(dplus1+1:end);

    dem1 = -y+x*w+ep*ones(n,1)+xi;
    dem2 = y-x*w+ep*ones(n,1)+xi;

    grad_w = t*[w(1:dplus1-1);0] -  sum( (dem1.^-1).*x)' + sum( (dem2.^-1).*x)';
    grad_xi = t*c*ones(n,1) - dem1.^-1 - dem2.^-1 - (xi.^-1);

    gradf = [grad_w;grad_xi];
end

