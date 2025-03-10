function [Dh, Dl, timers, S, scstat] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscaleFactor, numIters, useBinit)
% function [Dh, Dl, timers] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscaleFactor)
%
% Performs coupled dictionary training on paired high- and low-resolution image
%   patches (Xh, Xl, respectively).
%
% INPUTS:
%   Xh              = high-resolution image patches matrix; 
%                       dims=[patch_size^2,n_train_images]
%   Xl              = low-resolution image patches matrix 
%                       dims=[4*patch_size^2,n_train_images]
%   dict_size       = size of dict (# of atoms/columns)
%   lambda          = sparsity regularization
%   upscaleFactor   = zoom factor (e.g. 2 == increase res by factor 2)
%
% OUTPUTS:
%   Dh              = high res. patch dictionary
%   Dl              = low.res patch dictionary
%   timers          = struct with fields of elapsed times
%   S               = sparse codes from dictionary learning
%   scstat          = stats from sparse coding(sc)/dictionary training
%
% Called by: Demo_Dictionary_Training.m 

% % addpath(genpath('RegularizedSC'));
% addpath('RegularizedSC/sc2');
% addpath('RegularizedSC');

if nargin<7
    useBinit=false;
end

timers = [];
atic = tic; totcpu0 = cputime;

%[ dims of data
hDim = size(Xh, 1);
lDim = size(Xl, 1);

%[ (pre-)normalize Xh and Xl
hNorm = sqrt(sum(Xh.^2));
lNorm = sqrt(sum(Xl.^2));
Idx = find( hNorm & lNorm );
Xh = Xh(:, Idx);
Xl = Xl(:, Idx);
Xh = Xh./repmat(sqrt(sum(Xh.^2)), size(Xh, 1), 1);
Xl = Xl./repmat(sqrt(sum(Xl.^2)), size(Xl, 1), 1);

%[ joint learning of the dictionary
X = [sqrt(hDim)*Xh; sqrt(lDim)*Xl];
Xnorm = sqrt(sum(X.^2, 1));
clear Xh Xl;
%[ prune low-norm patches
X = X(:, Xnorm > 1e-5);
X = X./repmat(sqrt(sum(X.^2, 1)), hDim+lDim, 1);

Binit = [];
if useBinit
    idx = randperm(size(X, 2));   %included in orig code but not used...
    Binit = X(:,idx(1:dict_size));
end
%[ dictionary training
% fprintf('Running dictionary learning...\n');
[D,S,scstat] = reg_sparse_coding(X, dict_size, [], 0, lambda, numIters, [], Binit, false);
% g(X, num_bases, Sigma, beta, gamma, num_iters, batch_size, initB, saveDictEachIter)
Dh = D(1:hDim, :);
Dl = D(hDim+1:end, :);

% normalize the dictionary [was commented out in authors' code...]
% Dh = Dh./repmat(sqrt(sum(Dh.^2, 1)), hDim, 1);
% Dl = Dl./repmat(sqrt(sum(Dl.^2, 1)), lDim, 1);

patch_size = sqrt(size(Dh, 1));

atoc = toc(atic); totcpu = cputime - totcpu0;
timers.total = atoc;
timers.total_cpu = totcpu;

% %dont save dictionaries here...do in parent/calling function instead
% if 0 == 1 
%     dict_path = ['NewDictionary/D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
%     save(dict_path, 'Dh', 'Dl');
%     
%     dict_path = ['NewDictionary/S_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
%     save(dict_path, 'S');
%     
%     dict_path = ['NewDictionary/reg_s_c_stat_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
%     save(dict_path, 'scstat');
% end



