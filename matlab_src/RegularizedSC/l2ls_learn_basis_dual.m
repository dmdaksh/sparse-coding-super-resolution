function [B,options,results] = l2ls_learn_basis_dual(X, S, l2norm, Binit)
% [B,options,results] = l2ls_learn_basis_dual(X, S, l2norm, Binit)
%
% OUTPUTS:
%  B = dictionary
%  options = options used
%  results = struct with elap times, etc.
% 
% %%%%%%% [original notes below:] %%%%%%%%%%%%%%%%%%%%%
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
% EDIT: RJ 03-27-2024

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

options = optimoptions('fmincon','Algorithm','trust-region-reflective',...
    'SpecifyObjectiveGradient',true, 'Display','iter','HessianFcn','objective');

[x, fval, exitflag, output,lambda,grad,hessian] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);

% output.iterations
fval_opt = -0.5*N*fval;
fobjective_dual = fval_opt;
fobjective = fobjective_dual;
dual_lambda = x;

Bt = (SSt+diag(dual_lambda)) \ XSt';
B = Bt';

toc

results.fmincon.fval = fval;
results.fmincon.exitflag = exitflag;
results.fmincon.output = output;
results.fmincon.lambda = lambda;
results.fmincon.grad = grad;
results.fmincon.hessian = hessian;
results.fobjective = fobjective;
results.dual_lambda = dual_lambda;

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

