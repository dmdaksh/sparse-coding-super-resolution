function [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(X, B, S, Sigma, beta, gamma)
% [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(X, B, S, Sigma, beta, gamma)
% 
% INPUTS: 
%  X = image patch matrix
%  B = dictionary
%  S = sparse codes
%  Sigma, beta, gamma = some (regularization) parameters...
% 
% OUTPUTS:
%  fobj = objective value, where
%    fobj = fresidue + fsparsity + fregs

Err = X - B*S;

fresidue = 0.5*sum(sum(Err.^2));

fsparsity = gamma*sum(sum(abs(S)));

fregs = 0;
for ii = size(S, 2) %rj - changed size(S,1) to size(S,2), as they are indexing the ii column of S, not the ii row....
    fregs = fregs + beta*S(:, ii)'*Sigma*S(:, ii);
end

fobj = fresidue + fsparsity + fregs;