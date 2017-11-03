function [ hessf ] = hessb(x,y,ep,w_xi,t)
% Compute hessian of f
n = size(x,1);
dplus1 = size(x,2);
w = w_xi(1:dplus1);
xi = w_xi(dplus1+1:end);
dem1 = (-y+x*w+ep*ones(n,1)+xi).^-1;
dem2 = (y-x*w+ep*ones(n,1)+xi).^-1;

hww = t*[eye(dplus1-1), zeros(dplus1-1,1);
         zeros(1,dplus1-1),0] + sum(x'*(x .* dem1.^2))' + sum(x'*(x.* dem2.^2))';
hwxi = (x.*dem1.^2)' - (x.*dem2.^2)';
hxiw = hwxi';
hxixi = eye(n).*dem1.^2+eye(n).*dem2.^2+eye(n).*xi.^-2;

hessf = [hww, hwxi; 
        hxiw, hxixi];
end

