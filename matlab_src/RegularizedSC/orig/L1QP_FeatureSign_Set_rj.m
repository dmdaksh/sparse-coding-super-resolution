function [S] = L1QP_FeatureSign_Set_rj(X, B, Sigma, beta, gamma)

disp('..L1QP_FeatureSign_Set_rj..');

[dFea, nSmp] = size(X);
nBases = size(B, 2);

% sparse codes of the features
S = sparse(nBases, nSmp);
[Si,Sj,Ss] = find(S);

A = B'*B + 2*beta*Sigma;

for ii = 1:nSmp
%     disp(ii);
    if mod(ii,floor(nSmp/10))==0
        fprintf('[l1qp] sample %d/%d\n',ii,nSmp); 
    end
    b = -B'*X(:, ii);
%     [net] = L1QP_FeatureSign(gamma, A, b);
    temp = L1QP_FeatureSign_yang_rj(gamma, A, b);
    [i,j,s]=find(temp);
    
    Si = cat(1,Si,i);
    Sj = cat(1,Sj,j);
    Ss = cat(1,Ss,s);
    S = sparse(Si,Sj,Ss);
%     temp = sparse(i,j,s);
% 
%     S(1:dFea,ii) = temp; 
end