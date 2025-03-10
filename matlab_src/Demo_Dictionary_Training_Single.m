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

clear all; clc; close all;
addpath(genpath('RegularizedSC'));
addpath(genpath(pwd));


TR_IMG_PATH = 'Data/Training';

dict_size   = 512;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 5;            % image patch size
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 2;            % upscaling factor

% randomly sample image patches
[Xh, Xl] = rnd_smp_patch(TR_IMG_PATH, '*.bmp', patch_size, nSmp, upscaleFactor);

% prune patches with small variances, threshould chosen based on the
% training data
[Xh, Xl] = patch_pruning(Xh, Xl, 10);

% joint sparse coding 
ttic=tic;
[Dh, Sh, regscstat_h, dict_timers_h]  = train_single_dict_rj(Xh, dict_size, lambda, upscaleFactor); %, 'h');
ttoc=toc(ttic);
dict_timers_h.total_elap_time = ttoc;

fprintf(' -saving dictionary %s..\n',h);
dict_path = ['NewDictionary-Single/' 'h' '_D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '_s' num2str(upscaleFactor) '.mat' ];
save(dict_path, 'Dh', 'Sh', 'regscstat_h', "dict_timers_h");

% joint sparse coding 
ttic=tic;
[Dl, h_dict_timers] = train_coupled_dict_rj(Xh, Xl, dict_size, lambda, upscaleFactor, 'l');
ttoc=toc(ttic);
h_dict_timers.total_elap_time = ttoc;


% dict_path = ['Dictionary/D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '.mat' ];
% save(dict_path, 'Dh', 'Dl');

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


