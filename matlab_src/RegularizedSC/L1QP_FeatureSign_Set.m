function [S] = L1QP_FeatureSign_Set(X, B, Sigma, beta, gamma, verbose)
% [S] = L1QP_FeatureSign_Set(X, B, Sigma, beta, gamma, verbose)
%
% runs L1QP_FeatureSign_yang for each column of X
% INPUTS:
%   X = training data (matrix with patches as columns
%   B = dictionary
%   Sigma = [regularization parameter or something...]
%   beta  = [regularization parameter or something...]
%   gamma = passed to L1QP_FeatureSign_yang [regularization parameter or something...]
%   verbose (optional) = true to display text
% OUTPUTS:
%   S = sparse codes to recon X from B
%
% Edit: RJ 03-27-2024
%   Changed original code to fix indexing of sparse matrix S during update
%    of each column

if nargin<6, verbose=false; end

if verbose, disp('.. L1QP_FeatureSign_Set ..'); end

[dFea, nSmp] = size(X);
nBases = size(B, 2);

% sparse codes of the features
S = sparse(nBases, nSmp);
[Si,Sj,Ss] = find(S);

A = B'*B + 2*beta*Sigma;

for ii = 1:nSmp
    if verbose && mod(ii,floor(nSmp/10))==0
        fprintf('[l1qp] sample %d/%d\n',ii,nSmp); 
    end
    b = -B'*X(:, ii);
    [temp,l1qp_loss] = L1QP_FeatureSign_yang(gamma, A, b);
    % find indices of nonzeros
    [i,j,s]=find(temp);
    % update column of sparse S with nonzeros from temp
    Si = cat(1,Si,i);
    Sj = cat(1,Sj,j);
    Ss = cat(1,Ss,s);
    S = sparse(Si,Sj,Ss,nBases,nSmp);
end