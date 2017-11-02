clear; clc;

% load data
load('moviesGroups.mat');
load('moviesTest.mat');
load('moviesTrain.mat');
X = trainRatings;
y = trainLabels;
y(y == 0) = -1;

% params
n = length(y);
primal_dual_tol = 1e-6;
surr_dual_gap_tol = 2*1e-6;
alpha = 0.01;
beta = 0.5;
mu = 2;

m = 2 * n;
C = 1000;
epsilon = 1e-9;

K = get_K(X, y);

% get initial point
[u, v, w, lambda] = get_init_point(C, y);

t = 2*n / surr_dual_gap(u, v, w, C);

obj_vals = obj_fw(K, w)
barrier_obj_vals = barrier_obj(K, w, C, t)
while true
    t = t * mu;
    % compute newton step
    z = get_z(K, w, C, t, u, v, lambda, y);
    
    % find theta
    temp1 = -u ./ z(n+1:2*n);
    temp1 = temp1(z(n+1:2*n) < 0);
    temp2 = -v ./ z(2*n+1:3*n);
    temp2 = temp2(z(2*n+1:3*n) < 0);
    theta_max = min(1, min(temp1)); 
    theta_max = min(min(temp2), theta_max);
    theta = 0.99 * theta_max;
    
    % backtrack
    while true
        w_new = w + theta * z(1:n);
        u_new = u + theta * z(n+1:2*n);
        v_new = v + theta * z(2*n+1:3*n);
        lambda_new = lambda + theta * z(end);
        
        if any(w_new <= 0) && any(w_new  >= C) || norm(get_r(K, w_new, C, t, u_new, v_new, lambda_new, y)) > ...
                (1 - alpha * theta) * norm(get_r(K, w, C, t, u, v, lambda, y))
            theta = beta * theta;
        else
            break;
        end
    end
    
    % update params
    w = w + theta * z(1:n);
    u = u + theta * z(n+1:2*n);
    v = v + theta * z(2*n+1:3*n);
    lambda = lambda + theta * z(end);
    
    % check for convergence
    if get_prim_dual_tol(K, w, C, t, u, v, lambda, y) <= primal_dual_tol && ...
            surr_dual_gap(u, v, w, C) <= surr_dual_gap_tol
        break;
    end
    
    obj_vals = [obj_vals; obj_fw(K, w)];
    barrier_obj_vals = [barrier_obj_vals; barrier_obj(K, w, C, t)];
    obj_vals(end)
    barrier_obj_vals(end)
end

w(w < 1e-6) = 0;

save('res.mat', 'w');









