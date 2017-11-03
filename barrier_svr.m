%% Reading in data
x = csvread('X.csv',1,0);
y = csvread('y.csv',1,0);
% Center data
x = bsxfun(@minus, x, mean(x));
x = [x, ones(length(x),1)];

%% Set params
C = 10;
ep = 0.1;
n = size(x,1);
dplus1 = size(x,2);
alpha = 0.2;
beta = 0.9;
%t=10000;
%u=5;
t = 5;
u = 20;
m = 3*n;
fstar = 729.652;
gstar = 0.365;

[w_xi,sol] = initialize(x, y, ep);

i = 1;
thresh = 1e-9;
% Barrier Method loop
while m/t > thresh
    fprintf('===== iter %d, t = %d =====\n', i, t);
    assert( all(abs(y-x*w_xi(1:dplus1)) <= ep*ones(n,1)+w_xi(dplus1+1:end))==1);
    assert( all(w_xi(dplus1+1:end) >= zeros(n,1))==1);
    g(i) = 1/n*sum(y-x*w_xi(1:dplus1));
    %Newton's Method
    diff = ones(dplus1+n,1);
    while abs(diff) > thresh
        gradf = gradb(x,y,ep,w_xi,C,t);         %f''(x)
        hessf = hessb(x,y,ep,w_xi,t);           %f'(x)
        obj_val = bar_obj_val(x,y,ep,w_xi,t,C); %f(x)
        v = -pinv(hessf)*gradf;
        w_xi_update = w_xi + v;

        %Backtrack 
        step = 1;
        while bar_obj_val(x,y,ep,w_xi_update,t,C) > obj_val + alpha*step*gradf'*v
            step = beta*step;
            w_xi_update = w_xi + step*v;
            %fprintf(' ==> step: %d, backtrack cond: %f, obs_val_update: %f\n',step, bar_obj_val(x,y,ep,w_xi_update,t,C),obj_val + alpha*step*gradf'*v);
        end
        
        w_xi = w_xi_update;    
        f(i) = obj_val_orig(x,w_xi,C);
        g(i) = 1/n*sum(y-x*w_xi(1:dplus1));
        fprintf('f: %f, g: %f\n',f(i), g(i));
        diff = obj_val - bar_obj_val(x,y,ep,w_xi_update,t,C);
    end
    
    %Update params
    t = u*t;
    i = i+1;
end

        
% abs(y-x*w_xi_update(1:dplus1)) <= ep*ones(n,1)+w_xi_update(dplus1+1:end)
% abs(y-x*w_xi(1:dplus1)) <= ep*ones(n,1)+w_xi(dplus1+1:end)
      