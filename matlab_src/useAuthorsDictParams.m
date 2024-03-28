function [dict_size, lambda, patch_size, nSmp, upscaleFactor] = useAuthorsDictParams()
%  [dict_size, lambda, patch_size, nSmp, upscaleFactor] = useAuthorsDictParams()
% 
% returns dictionary parameters from original author code

dict_size   = 512;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 5;            % image patch size
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 2;            % upscaling factor


