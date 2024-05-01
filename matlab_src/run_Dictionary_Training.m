function [Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
run_Dictionary_Training( indir, outdir, dict_size, patch_size, ...
    lambda, nSmp, upscaleFactor, numIters, pruningVarThresh )
% ========================================================================
% Function for running dictionary training demo
%   Modified from authors' code
%   
% Modified to function-form from Demo_Dictionary_Training.m 
% =========================================================================

% %[ ensure you are in /matlab_src
%  %%% not adding here... %%%

%% Setup
% %[ add dirs to path
% % addpath(genpath(pwd));
% addpath('RegularizedSC/sc2');
% addpath('RegularizedSC');
% addpath('utils');
% addpath('qtfm');
% 
% %[ location of trianing images
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% % indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';
% 
% %[ location to save dictionary
% outdir = 'Dictionary_rj_2';
% 
% %[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
% dict_size   = 1024;          % dictionary size
% lambda      = 0.1;         % sparsity regularization
% patch_size  = 3;            % image patch size
% patch_overlap = patch_size - 1;
% nSmp        = 100000;       % number of patches to sample
% upscaleFactor     = 3;            % upscaling factor
% numIters = 10;
% 
% %[ other parameters
% pruningVarThresh = 10;

if ~exist(outdir,'dir')
    fprintf('---making output dir:\n %s\n',outdir);
    mkdir(outdir);
end

%% Get training data (patches)
%[ randomly sample image patches
d = dir([indir filesep '*.*']);
[~,~,fext] = fileparts(d(end).name);
if ~contains(fext,'jpg') && ~contains(fext,'bmp')
    fext = '.bmp';
    d = dir([indir filesep '*.bmp']);
    if isempty(d)
        fext = '.jpg';
    end
end
fext = strcat('*',fext);

switch fext
    case '*.bmp'
        trainsetName = 'Yang';
    case '*.jpg'
        trainsetName = 'BSD300';
end

%% Sample patches
fprintf(' -sampling patches..\n');
[Xh, Xl] = rnd_smp_patch(indir, fext, patch_size, nSmp, upscaleFactor);
%[ prune patches with small variances, threshould chosen based on the
[Xh, Xl] = patch_pruning(Xh, Xl, pruningVarThresh);

%% Dictionary learning
fprintf(' -training coupled dict..\n');
% joint sparse coding 
dictic=tic; diccpu = cputime;
[Dh, Dl, dict_timers, S, sparsecode_stat] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscaleFactor, numIters, false);
dict_timers.elaptime = toc(dictic);
dict_timers.cputime = cputime - diccpu;

%% Save dictionary
fprintf(' -saving dict..\n');

dlparams.dict_size   = dict_size;          % dictionary size
dlparams.lambda      = lambda;         % sparsity regularization
dlparams.patch_size  = patch_size;            % image patch size
dlparams.nSmp        = nSmp;       % number of patches to sample
dlparams.upscaleFactor     = upscaleFactor;            % upscaling factor
dlparams.numIters = numIters;
dlparams.pruningVarThresh = pruningVarThresh;
dlparams.trainsetName = trainsetName;

dict_name = ['D_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
    '_zoom-' num2str(upscaleFactor) '_' trainsetName '.mat'];
dict_path = fullfile(outdir,dict_name);
fprintf('[dict_dir: %s \n',outdir);
fprintf('[dict_name: %s \n',dict_name);
save(dict_path, 'Dh', 'Dl', 'dict_timers','dlparams', 'S', 'sparsecode_stat');

dlparams.dict_path = dict_path;


if 0 == 1
    fprintf(' -making plots..\n');
    
    %[ display dictionary atoms
    figure('position',[248 356 1222 510],'color','w');
    tiledlayout(1,2);
    display_network_nonsquare_subplots(Dl,patch_size*2);
    title('$D_l$ patches','FontSize',25,'Interpreter','latex');
    display_network_nonsquare_subplots(Dh,patch_size);
    title('$D_h$ patches','FontSize',25,'Interpreter','latex');
    drawnow;
    fig_name = ['show-atoms_Dh,Dl_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
        '_zoom-' num2str(upscaleFactor) '.png'];
    fig_path = fullfile(outdir,fig_name);
    print(gcf,fig_path,'-dpng');

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
    fig_name = ['show-atoms_Dh,Dl-sep_' num2str(dict_size) '_lam-' num2str(lambda) '_patchsz-' num2str(patch_size) ...
        '_zoom-' num2str(upscaleFactor) '.png'];
    fig_path = fullfile(outdir,fig_name);
    print(gcf,fig_path,'-dpng');
    

%     %[ Analyze dict training results
%     res = load('NewDictionary/reg_s_c_stat_512_0.15_5_s2.mat');
%     niters = length(res.regscstat.fobj_avg);
%     
%     %[ plot dict training results/stats
%     figure('color','w','position',[182 492 1282 244]);
%     tiledlayout(1,4);
%     
%     nexttile
%     p1 = plot(1:niters, res.regscstat.fobj_avg,'LineWidth',1.5,'Marker','^');
%     title('DL Objective Function');
%     xlabel('Iteration');
%     ylabel('Objective value');
%     set(gca,'FontSize',12);
%     
%     nexttile
%     p2 = plot(1:niters, 100*res.regscstat.sparsity,'LineWidth',1.5,'Marker','^');
%     title('Codebook coefficient sparsity');
%     xlabel('Iteration');
%     ylabel('Sparsity level (% nonzero)');
%     set(gca,'FontSize',12);
%     
%     nexttile
%     hold on;
%     p3a = plot(1:niters, res.regscstat.stime,'LineWidth',1.5,'Marker','^');
%     p3b = plot(1:niters, res.regscstat.btime,'LineWidth',1.5,'Marker','v');
%     title('Computation time');
%     xlabel('Iteration');
%     ylabel('Elapsed time (s)');
%     legend('Sparse code update','Dictionary update', ...
%         'location','best','fontsize',12);
%     set(gca,'FontSize',12);
%     
%     nexttile
%     p4 = plot(1:niters, cumsum(res.regscstat.elapsed_time),'LineWidth',1.5,'Marker','v');
%     title({'Cumulative elapsed time',sprintf('Total time = %.1f min',sum(res.regscstat.elapsed_time)/60)});
%     xlabel('Iteration');
%     ylabel('Elapsed time (s)');
%     set(gca,'FontSize',12);
%     
%     print(gcf,'NewDictionary/DictTrain-Testing-Results_plots.png','-dpng');

end

