clear;
close all;

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
t=[5, 10000];
u=[20, 5];

m = 3*n;
fstar = 729.652;
gstar = 0.365;

thresh = 1e-9;
% Barrier Method loop
for j = 1 : 2
   
    [w_xi,sol] = initialize(x, y, ep);
     w_xi(1:dplus1) = 0;
     w_xi(dplus1+1:end) = y;

    i = 1;
    while m/t(j) > thresh
        assert( all(abs(y-x*w_xi(1:dplus1)) <= ep*ones(n,1)+w_xi(dplus1+1:end))==1);
        assert( all(w_xi(dplus1+1:end) >= zeros(n,1))==1);
        
        %Newton's Method
        diff = ones(dplus1+n,1);
        while abs(diff) > thresh
            gradf = gradb(x,y,ep,w_xi,C,t(j));         %f''(x)
            hessf = hessb(x,y,ep,w_xi,t(j));           %f'(x)
            obj_val = bar_obj_val(x,y,ep,w_xi,t(j),C); %f(x)
            v = -pinv(hessf)*gradf;
            w_xi_update = w_xi + v;

            %Backtrack 
            step = 1;
            while bar_obj_val(x,y,ep,w_xi_update,t(j),C) > obj_val + alpha*step*gradf'*v
                step = beta*step;
                w_xi_update = w_xi + step*v;
            end

            w_xi = w_xi_update;    
            if j == 1
                f1(i) = obj_val_orig(x,w_xi,C);
                g1(i) = 1/n*sum((y-x*w_xi(1:dplus1)).^2);
                fprintf('f: %f, g: %f\n',f1(i), g1(i));
            else
                f2(i) = obj_val_orig(x,w_xi,C);
                g2(i) = 1/n*sum((y-x*w_xi(1:dplus1)).^2);
                fprintf('f: %f, g: %f\n',f2(i), g2(i));
            end
            diff = obj_val - bar_obj_val(x,y,ep,w_xi_update,t(j),C);
            i = i+1;
        end

        %Update params
        t(j) = u(j)*t(j);
    end
end

figure(1)
hold on;
title('f vs. fstar','FontSize',14);
xlabel('Iteration(n)','FontSize',14);
ylabel('log(f-fstar)','FontSize',14);
plot(1:length(f1), log(f1-fstar),'-','LineWidth',2)
plot(1:length(f2), log(f2-fstar),'-','LineWidth',2)
% plot(1:length(g1), log(g1-gstar),'-','LineWidth',2)
legend('(t = 5, u = 20)','(t = 10000, u = 5)','FontSize',14);

figure(2)
hold on;
title('g vs. gstar','FontSize',14);
xlabel('Iteration(n)','FontSize',14);
ylabel('log(g-gstar)','FontSize',14);
semilogy(1:length(g1), log(g1-gstar),'-','LineWidth',2)
semilogy(1:length(g2), log(g2-gstar),'-','LineWidth',2)
legend('(t = 5, u = 20)','(t = 10000, u = 5)','FontSize',14);
