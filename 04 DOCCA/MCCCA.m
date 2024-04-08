function [u, v, obj] = MCCCA(X, Y, opts)
%% Problem
%
%  max  u' X 'Y v - 1/2*alpha1*||Xu||^2 - 1/2*alpha2*||Yv||^2 -
%  lambda1*||u||_FGL - lambda2*||v||_GGL
% --------------------------------------------------------------------

p = size(X,2);
q = size(Y,2);

% Calculate coverance within X and Y
XX = X'*X;
YY = Y'*Y;
alpha = 2;
LX = get_connectivity(X',alpha);
LY = get_connectivity(Y',alpha);
Lu = get_connectivity(X,alpha);
Lv = get_connectivity(Y,alpha);

% initialize w1 here
u0 = ones(p, 1);
% --------------------
% initialize w2 here
v0 = ones(q, 1);

u = u0;
v = v0;
% % scale u and v
scale = sqrt(u'*XX*u);
u = u./scale;
scale = sqrt(v'*YY*v);
v = v./scale;

% set parameters
alpha1 = opts.alpha1;
alpha2 = opts.alpha2;
lambda1 = opts.lambda1;
lambda2 = opts.lambda2;
beta1 =  opts.beta1;
beta2 =  opts.beta2;
gamma1 = opts.gamma1;
gamma2 = opts.gamma2;


% set stopping criteria
max_Iter = 100;
i = 0;
tol = 1e-5;
obj = [];
tv = inf;
tu = inf;

while (i<max_Iter && (tu>tol || tv>tol)) % default 100 times of iteration
    i = i+1;
    
    % update u
    % -------------------------------------
    % update diagnal matrix Du
    Du = updateD2(u) / 2;
    % solve u
    u_old = u;
    F1 = lambda2*X'*LX*X+(1+alpha2)*XX+gamma2*Lu*u;
    Yv = Y*v;
    b1 = X'*Yv;
    u = F1\b1;
    % scale u
    scale = sqrt(u'*XX*u);
    u = u./scale;
    
    % update v
    % -------------------------------------
    % update diagnal matrix D2
    Dv = updateD2(v) / 2;
    % solve v
    v_old = v;
    F2 = lambda1*Y'*LY*Y+(1+alpha1)*YY+gamma1*Lv*v;
    Xu = X*u;
    b2 = Y'*Xu;
    v = F2\b2;
    % scale v
    scale = sqrt(v'*YY*v);
    v = v./scale;
    
    % stopping condition
    % -------------------------
    if i > 1
        tu = max(abs(u-u_old));
        tv = max(abs(v-v_old));
    else
        tu = tol*10;
        tv = tol*10;
    end
    obj(end+1) = -u'*X'*Y*v+lambda1*v'*Y'*LY*Y*v+lambda2*u'*X'*LX*X*u+gamma1*v'*Lv*v+gamma2*u'*Lu*u+alpha2*(u'*XX*u-1)+alpha1*(v'*YY*v-1);
end
end