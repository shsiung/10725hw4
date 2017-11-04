function [ gradf ] = gradb(x,y,ep,w_xi,c,t)
%find the gradient of f
    n = size(x,1);
    dplus1 = size(x,2);
    w = w_xi(1:dplus1);
    xi = w_xi(dplus1+1:end);

    dem1 = (-y+x*w-ep*ones(n,1)-xi).^-1;
    dem2 = (y-x*w-ep*ones(n,1)-xi).^-1;

    term1 = 0;
    term2 = 0;
    for i = 1 : n
        term1 = term1 + dem1(i)*x(i,:);
        term2 = term2 + dem2(i)*x(i,:);
    end
    
    grad_w = t*[w(1:dplus1-1);0] -  term1' + term2';
    grad_xi = t*c*ones(n,1) + dem1 + dem2 - (xi.^-1);

    gradf = [grad_w;grad_xi];
end

