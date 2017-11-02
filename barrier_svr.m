%% Reading in data
x = csvread('X.csv',1,0);
y = csvread('y.csv',1,0);

% Center data
x = bsxfun(@minus, x, mean(x));
%% Set params
C = 10;
ep = 0.1;
n = size(x,1);
dplus1 = size(x,2);
alpha = 0.2;
beta = 0.9;
%t=10000
%u=5
t = 5;
u = 20;
m = 10*n;
fstar = 729.652;
gstar = 0.365;

[w_xi,sol] = initialize(x, y, ep);

i = 1;
thresh = 1e-9;
% Barrier Method loop
while m/t > thresh
    fprintf('===== iter %d =====\n', i);
    %Newton's Method
    diff = ones(dplus1+n,1);
    while norm(diff) > thresh
        gradf = gradb(x,y,ep,w_xi,C,t);         %f''(x)
        hessf = hessb(x,y,ep,w_xi,t);           %f'(x)
        obj_val = bar_obj_val(x,y,ep,w_xi,t,C); %f(x)
        v = -pinv(hessf)*gradf;
        w_xi_update = w_xi + v;
        
      % abs(y-x*w_xi_update(1:dplus1)) <= ep*ones(n,1)+w_xi_update(dplus1+1:end)
      % abs(y-x*w_xi(1:dplus1)) <= ep*ones(n,1)+w_xi(dplus1+1:end)
      
        %Backtrack 
        step = 1;
        obj_val_update = bar_obj_val(x,y,ep,w_xi_update,t,C); %f(x+step*v)
        backtrack_cond = obj_val + alpha*step*gradf'*v;       %f(x) + alpha*t*hess*update
        while obj_val_update > backtrack_cond
            step = beta*step;
            w_xi_update = w_xi + step*v;
            obj_val_update = bar_obj_val(x,y,ep,w_xi_update,t,C);
            backtrack_cond = obj_val + alpha*step*gradf'*v; 
      %      fprintf(' ==> step: %d, backtrack cond: %f, obs_val_update: %f\n',step, backtrack_cond,obj_val_update);
        end

        w_xi_last = w_xi;
        w_xi = w_xi + step * v;
       % fprintf('   update obj_barrier:%f\n', obj_val);
        diff = w_xi - w_xi_last;
    end
    
    f(i) = obj_val_orig(x,w_xi,C);
    fprintf('objective value (original) is: %f\n', f(i));
    
    %Update params
    t = u*t;
    i = i+1;
end