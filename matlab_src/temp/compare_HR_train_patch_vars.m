function compare_HR_train_patch_vars()

if ~isdeployed
addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');
end

%[ location to save dictionary
outdir = pwd; %'temp';

%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 1024;          % dictionary size
lambda      = 0.1;         % sparsity regularization
patch_size  = 3;            % image patch size
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 3;            % upscaling factor

%[ other parameters
pruningVarThresh = 10;

%% sample patches
jindir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';
bindir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
[bXh, bXl] = rnd_smp_patch(bindir, '*.bmp', patch_size, nSmp, upscaleFactor);
[jXh, jXl] = rnd_smp_patch(jindir, '*.jpg', patch_size, nSmp, upscaleFactor);
%% histograms of var(Xh)

figure('color','w','position',[497 498 974 297]); 
subplot(121);
hold on;  
histogram(var(bXh),0:10:500); %1000);  1000);  
histogram(var(jXh),0:10:500); %1000);  1000);
legend('Yang','BSD300');
title('Xh patch variance histogram');
xlabel('var(patch)');
ylabel('# of patches');
set(gca,'FontSize',16);
subplot(122);
hold on;  
histogram(var(bXl),0:10:500); %1000);  
histogram(var(jXl),0:10:500); %1000);
legend('Yang','BSD300');
title('Xl patch variance histogram');
xlabel('var(patch)');
ylabel('# of patches');
set(gca,'FontSize',16);
print(gcf,fullfile(outdir,'Yang-vs-BSD300_patch_var_comparison_histogram.png'),'-dpng');

