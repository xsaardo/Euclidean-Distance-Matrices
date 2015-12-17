clc; clear; close all;

%% Generate Initial Point Set
n = 20; % Number of initial points
d = 2; % dimension or data set
X = rand(d,n,'single')*10; % 2xn matrix

%% Generate EDM from Point Set
edm_X = ones(n,1)*diag(X'*X).' - 2*(X'*X) + diag(X'*X)*ones(1,n);

% Geometric Centering
J = eye(n)-ones(n)/n;
gram = -0.5*J*edm_X*J;

% Eigenvalue Decomposition
[U, L] = eig(gram);

L = sort(diag(L),'descend');
L = L(1:d);

% Reconstruction
X_hat = [sqrt(diag(L)) zeros(d,n-d)]*U';

%% Orthogonal Procrustes (Rotating reconstructed data set)
num_anchors = 4;

Xa = X_hat(:,1:num_anchors);
Y = X(:,1:num_anchors);
Xa_mean = 1/num_anchors*sum(Xa,2);
Y_mean = 1/num_anchors*sum(Y,2);

Xa = Xa - Xa_mean*ones(1,num_anchors);
Y = Y - Y_mean*ones(1,num_anchors);

[U, S, V] = svd(Xa*Y');
R = V*U';

X_hat = R*(X_hat - Xa_mean*ones(1,n))+Y_mean*ones(1,n);

scatter(X(1,:),X(2,:));
figure;
scatter(X_hat(1,:),X_hat(2,:));