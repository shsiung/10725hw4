%% Reading in data
x = csvread('X.csv',1,0);
y = csvread('y.csv',1,0);
x = bsxfun(@minus, x, mean(x));
%% Set params
C = 10;
ep = 0.1;
t = 5;
u = 20;

[w, xi,sol] = initialize(x, y, ep);
