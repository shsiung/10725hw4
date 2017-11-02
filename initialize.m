function [w, xi,sol] = initialize(x, y, ep)
% Initialize interior points

n = length(y);
dplus1 = size(x,2);

A = [-x,              -eye(n), -eye(n)    ,zeros(n,n),zeros(n,n);
      x,              -eye(n), zeros(n,n) ,-eye(n)   ,zeros(n,n);
      zeros(n,dplus1),-eye(n), zeros(n,n) ,zeros(n,n),-eye(n)  ];

b = [ ep*ones(n,1) - y;
      ep*ones(n,1) + y;
      zeros(n,1)     ];

Aeq = [];
beq = [];

ub =  Inf + ones(dplus1+4*n,1);
lb = [-Inf + ones(dplus1+n,1);
       zeros(2*n,1)];

f = [zeros(dplus1+n,1);
     ones(3*n,1)];
sol = linprog(f, A, b, Aeq, beq, lb, ub);
w = sol(1:dplus1);
xi = sol(dplus1+1:dplus1+n);
end

