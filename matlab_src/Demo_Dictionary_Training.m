% ========================================================================
% Script for running dictionary training demo
%   Modified from authors' code
%   Uses images provided by authors for trianing
%   
% ========================================================================
% Original author code downloaded from:
%  https://github.com/tingfengainiaini/sparseCodingSuperResolution
% ========================================================================
% Demo codes for dictionary training by joint sparse coding
% 
% Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================

clear; clc; close all;

%% Setup
%[ add dirs to path
% addpath(genpath(pwd));
addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

%[ location of trianing images
indir = 'Data/Training';
%[ location to save dictionary
outdir = 'Dictionary_rj';
if ~exist(outdir,'dir')
    fprintf('---making output dir:\n %s\n',outdir);
    mkdir(outdir);
end

%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 2048;          % dictionary size
lambda      = 0.1;         % sparsity regularization
patch_size  = 3;            % image patch size
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 3;            % upscaling factor
numIters = 10;

%[ other parameters
pruningVarThresh = 10;

%% Get training data (patches)
%[ randomly sample image patches
[Xh, Xl] = rnd_smp_patch(indir, '*.bmp', patch_size, nSmp, upscaleFactor);
%[ prune patches with small variances, threshould chosen based on the
[Xh, Xl] = patch_pruning(Xh, Xl, pruningVarThresh);

%% Dictionary learning
% joint sparse coding 
tcdtic=tic;
[Dh, Dl, dict_timers, S, sparsecode_stat] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscaleFactor, numIters);
tcdtoc=toc(tcdtic);
dict_timers.total_elap_time = tcdtoc;

%% Save dictionary
dlparams.dict_size   = dict_size;          % dictionary size
dlparams.lambda      = lambda;         % sparsity regularization
dlparams.patch_size  = patch_size;            % image patch size
dlparams.nSmp        = nSmp;       % number of patches to sample
dlparams.upscaleFactor     = upscaleFactor;            % upscaling factor
dlparams.numIters = numIters;
dlparams.pruningVarThresh = pruningVarThresh;


% dict_name = ['D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '.mat'];
dict_name = ['D_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
    '_zoom-' num2str(upscaleFactor) '.mat'];
dict_path = fullfile(outdir,dict_name);
save(dict_path, 'Dh', 'Dl', 'dict_timers','dlparams', 'S', 'sparsecode_stat');

if 0 == 1
    %% Load results
    load('NewDictionary/D_512_0.15_5_s2.mat','Dh','Dl');
    load('NewDictionary/S_512_0.15_5_s2.mat','S');
    
    %[ display dictionary atoms
    figDl = display_network_nonsquare2(Dl,5);
    figDh = display_network_nonsquare2(Dh);
    
    %[ Analyze dict training results
    res = load('NewDictionary/reg_s_c_stat_512_0.15_5_s2.mat');
    niters = length(res.regscstat.fobj_avg);
    
    %[ plot dict training results/stats
    figure('color','w','position',[182 492 1282 244]);
    tiledlayout(1,4);
    
    nexttile
    p1 = plot(1:niters, res.regscstat.fobj_avg,'LineWidth',1.5,'Marker','^');
    title('DL Objective Function');
    xlabel('Iteration');
    ylabel('Objective value');
    set(gca,'FontSize',12);
    
    nexttile
    p2 = plot(1:niters, 100*res.regscstat.sparsity,'LineWidth',1.5,'Marker','^');
    title('Codebook coefficient sparsity');
    xlabel('Iteration');
    ylabel('Sparsity level (% nonzero)');
    set(gca,'FontSize',12);
    
    nexttile
    hold on;
    p3a = plot(1:niters, res.regscstat.stime,'LineWidth',1.5,'Marker','^');
    p3b = plot(1:niters, res.regscstat.btime,'LineWidth',1.5,'Marker','v');
    title('Computation time');
    xlabel('Iteration');
    ylabel('Elapsed time (s)');
    legend('Sparse code update','Dictionary update', ...
        'location','best','fontsize',12);
    set(gca,'FontSize',12);
    
    nexttile
    p4 = plot(1:niters, cumsum(res.regscstat.elapsed_time),'LineWidth',1.5,'Marker','v');
    title({'Cumulative elapsed time',sprintf('Total time = %.1f min',sum(res.regscstat.elapsed_time)/60)});
    xlabel('Iteration');
    ylabel('Elapsed time (s)');
    set(gca,'FontSize',12);
    
    print(gcf,'NewDictionary/DictTrain-Testing-Results_plots.png','-dpng');

end

