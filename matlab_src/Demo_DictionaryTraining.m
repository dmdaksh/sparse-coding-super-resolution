% Script to run dictionary training for ScSR
%
% Calls: 
%   run_Dictionary_Training.m 
%                   [randomly samples patches, runs dictionary training, 
%                   and saves learned dictionaries as .mat files]
%
% RJ | 04-2024 | EECS 556 W24 Project | Group 8

% Change the path below to the location of your /matlab_src directory!!!
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

% Adds necessary paths
addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% %[ location of trianing images
indir = 'Data/Training';

% %[ location to save dictionaries to
dictdir = 'test_Dictionaries';
if ~exist(dictdir,'dir'), mkdir(dictdir); end

%[ Parameters to use for dictionary training
patch_size = 5;                 % image patch size
lambda = 0.1;                   % sparsity regularization parameter
overlap = patch_size - 1;       % overlap between adjacent patches
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxDLIter = 10;                   % # of DL iterations (10 seems to converge)
pruningVarThresh = 10;          % for pruning randomly sampled patches
nbPatches = 100000;             % number of randomly sampled image patches
dictSize = 1024;                % number of dictionary atoms (cols/size of dict)

% run dict training (and time how long it takes)
[Dh, Dl, dict_timers, dlparams, S, sparsecode_stat, Xh, Xl ] = ...
    run_Dictionary_Training( indir, dictdir, dictSize, patch_size, ...
    lambda, nbPatches, up_scale, maxDLIter, pruningVarThresh );    

% the name of the output dicts (saved during run_Dictionary_Training.m)
fprintf(' --Training complete - dicts saved to:\n %s\n',dlparams.dict_path);

    
%[ display dictionary atoms
figure('position',[248 356 1222 510],'color','w');
tiledlayout(1,2);
display_network_nonsquare_subplots(Dl,patch_size*2);
title('$D_l$ patches','FontSize',25,'Interpreter','latex');
display_network_nonsquare_subplots(Dh,patch_size);
title('$D_h$ patches','FontSize',25,'Interpreter','latex');
drawnow;
% fig_name = ['show-atoms_Dh,Dl_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%     '_zoom-' num2str(upscaleFactor) '.png'];
% fig_path = fullfile(outdir,fig_name);
% print(gcf,fig_path,'-dpng');

%[ separate 4 features from each patch of low-res dict
Dl_sep = reshape(Dl,size(Dh,1),[]);
%[ display dictionary atoms
figure('position',[248 356 1222 510],'color','w');
tiledlayout(1,2);
display_network_nonsquare_subplots(Dl_sep,patch_size);
title('$D_l$ patches','FontSize',25,'Interpreter','latex');
display_network_nonsquare_subplots(Dh,patch_size);
title('$D_h$ patches','FontSize',25,'Interpreter','latex');
drawnow;
% fig_name = ['show-atoms_Dh,Dl-sep_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
%     '_zoom-' num2str(upscaleFactor) '.png'];
% fig_path = fullfile(outdir,fig_name);
% print(gcf,fig_path,'-dpng');
