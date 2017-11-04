function [ hessf ] = hessb(x,y,ep,w_xi,t)
% Compute hessian of f
n = size(x,1);
dplus1 = size(x,2);
w = w_xi(1:dplus1);
xi = w_xi(dplus1+1:end);
dem1 = (-y+x*w-ep*ones(n,1)-xi).^-2;
dem2 = (y-x*w-ep*ones(n,1)-xi).^-2;

term1 = zeros(17,17);
term2 = zeros(17,17);
term3 = x;
term4 = x;
term5 = eye(n);
term6 = eye(n);
term7 = eye(n);
xi2 = xi.^-2;
for i = 1 : n
    term1 = term1 + x(i,:)'*x(i,:)*dem1(i);
    term2 = term2 + x(i,:)'*x(i,:)*dem2(i);
    term3(i,:) = x(i,:)*dem1(i);
    term4(i,:) = x(i,:)*dem2(i);
    term5(i,i) = dem1(i);
    term6(i,i) = dem2(i);
    term7(i,i) = xi2(i);
end

hww = t*[eye(dplus1-1), zeros(dplus1-1,1);
         zeros(1,dplus1-1),0] + term1' + term2';
hwxi = -(term3)' + (term4)';
hxiw = hwxi';
hxixi = term5+term6+term7;

hessf = [hww, hwxi; 
        hxiw, hxixi];
end

