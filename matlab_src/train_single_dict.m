function [D, S, regscstat, timers] = train_single_dict_rj(X, dict_size, lambda, upscaleFactor, nameTag)

addpath(genpath('RegularizedSC'));

timers = [];
atic = tic;

hDim = size(X, 1);
% % lDim = size(Xl, 1);
% 
% % should pre-normalize Xh and Xl !
hNorm = sqrt(sum(X.^2));
% % lNorm = sqrt(sum(Xl.^2));
Idx = find( hNorm ); %& lNorm );
% 
X = X(:, Idx);
% % Xl = Xl(:, Idx);
% 
X = X./repmat(sqrt(sum(X.^2)), size(X, 1), 1);
% % Xl = Xl./repmat(sqrt(sum(Xl.^2)), size(Xl, 1), 1);
% 
% % joint learning of the dictionary
% X = [sqrt(hDim)*Xh]; %; sqrt(lDim)*Xl];
Xnorm = sqrt(sum(X.^2, 1));
% 
% clear Xh; % Xl;
% 
X = X(:, Xnorm > 1e-5);
% X = X./repmat(sqrt(sum(X.^2, 1)), hDim, 1);
% 
% idx = randperm(size(X, 2));

% dictionary training
disp('reg_sparse_coding')
sctic = tic;
[D,S,regscstat] = reg_sparse_coding_rj(X, dict_size, [], 0, lambda, 40);
sctoc = toc(sctic);
fprintf('Elapsed time: %g s\n',sctoc);

% Dh = D;
% D = Dh(1:hDim, :);
% % Dl = D(hDim+1:end, :);

% normalize the dictionary
% Dh = Dh./repmat(sqrt(sum(Dh.^2, 1)), hDim, 1);
% Dl = Dl./repmat(sqrt(sum(Dl.^2, 1)), lDim, 1);

patch_size = sqrt(size(D, 1));
regscstat.patch_size = patch_size;

atoc = toc(atic);

timers.sparsecoding = sctoc;
timers.total = atoc;

if nargin<5 || isempty(nameTag)
    return
else
    fprintf(' -saving dictionary %s..\n',nameTag);
    dict_path = ['NewDictionary-Single/' nameTag '_D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
    save(dict_path, 'D', 'S', 'regscstat', "timers");
end

% dict_path = ['NewDictionary-Single/S_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
% save(dict_path, 'S');
% 
% dict_path = ['NewDictionary-Single/reg_s_c_stat_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
% save(dict_path, 'regscstat');




