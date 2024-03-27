function B = l2ls_learn_basis_dual(X, S, l2norm, Binit)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_B   0.5*||X - B*S||^2
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

L = size(X,1);
N = size(X,2);
M = size(S, 1);

tic
SSt = S*S';
XSt = X*S';

if exist('Binit', 'var')
    dual_lambda = diag(Binit\XSt - SSt);
else
    dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
end

c = l2norm^2;
trXXt = sum(sum(X.^2));

lb=zeros(size(dual_lambda));

options = optimset('GradObj','on');

% options = optimset('GradObj','on', 'Hessian','on');

% HFun = @(x) fobj_basis_dual_h(x, SSt, XSt, X, c, trXXt);
% options = optimset('GradObj','on', 'Hessian','on','HessFcn',HFun);

% options = optimset('GradObj','on', 'Hessian','on','HessianFcn',@(x) fobj_basis_dual_h(x, SSt, XSt, X, c, trXXt));

%  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);

% options = optimoptions('fmincon','Algorithm','interior-point',...
%     'SpecifyObjectiveGradient',true,'HessianFcn','objective');

% options = optimoptions('fmincon','Algorithm','interior-point',...
%     'SpecifyObjectiveGradient',true, 'Display','iter-detailed'); %,'HessianFcn',@(x) fobj_basis_dual_h(x, SSt, XSt, X, c, trXXt));

options = optimoptions('fmincon','Algorithm','trust-region-reflective',...
    'SpecifyObjectiveGradient',true, 'Display','iter-detailed','HessianFcn','objective');



disp('fmincon');

% [x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);

[x, fval, exitflag, output,lambda,grad,hessian] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);

% [x, fval, exitflag, output,lambda,grad,hessian] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), ...
%     dual_lambda, [], [], [], [], lb, [], [], options);

% output.iterations
fval_opt = -0.5*N*fval;
dual_lambda= x;

Bt = (SSt+diag(dual_lambda)) \ XSt';
B_dual= Bt';
fobjective_dual = fval_opt;


B= B_dual;
fobjective = fobjective_dual;
toc

return;

% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,g,H] = fobj_basis_dual(dual_lambda, SSt, XSt, X, c, trXXt)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
    if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
    end
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Hout = fobj_basis_dual_Hessian(x,lambda)
% Compute the objective function value at x

% SSt_inv = inv(SSt + diag(x));
% temp = XSt*SSt_inv;
temp = XSt/(SSt + diag(x));
% Hessian evaluated at x
% H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
H = -2.*((temp'*temp).*SSt_inv);
Hout = -H;
    

return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = fobj_basis_dual_g(dual_lambda, SSt, XSt, X, c, trXXt)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

% if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
%     if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
%         H = -2.*((temp'*temp).*SSt_inv);
%         H = -H;
%     end
% end
return

function [H] = fobj_basis_dual_h(dual_lambda, SSt, XSt, X, c, trXXt)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

% if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
%     if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
%     end
% end
return


function [H] = fobj_basis_dual_h2(dual_lambda)
% Compute the objective function value at x
L= size(XSt,1);
M= length(dual_lambda);

SSt_inv = inv(SSt + diag(dual_lambda));

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
end
f= -f;

% if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g = zeros(M,1);
    temp = XSt*SSt_inv;
    g = sum(temp.^2) - c;
    g= -g;
    
    
%     if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H = -2.*((temp'*temp).*SSt_inv);
        H = -H;
%     end
% end
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

