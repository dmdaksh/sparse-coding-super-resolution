
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% %[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% % indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';
% 
% %[ location to save dictionary
% outdir = 'Dictionary_rj_mod_feat3';
% if ~exist(outdir,'dir'), mkdir(outdir); end
% 
% %% zoom 3, patch 3
% 
% %[ location of trianing images
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% % indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';
% 
% %[ location to save dictionary
% outdir = 'Dictionary_rj_mod';
% if ~exist(outdir,'dir'), mkdir(outdir); end
% 
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
% % 
% % profile on;
% % 
% % DLtic = tic; DLcpu = cputime;
% % 
% % [Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
% % run_Dictionary_Training( indir, outdir, dict_size, patch_size, ...
% %     lambda, nSmp, upscaleFactor, numIters, pruningVarThresh );
% % 
% % DLtimer.elaptime = toc(DLtic);
% % DLtimer.cputime = cputime - DLcpu;
% % 
% % [foutdir,foutname,~] = fileparts(dlparams.dict_path);
% % prof_dir = [foutdir filesep 'DictProfiler-' foutname];
% % 
% % profinfo = profile('info');
% % profsave(profinfo,prof_dir);




outdir = 'New-Results-dict-compare';
if ~exist(outdir,'dir'), mkdir(outdir); end

dictroot = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/newdicts';

samplesizes = [25,50,100];
nsamplesizes = length(samplesizes);

samplesizesString = {'25','50','100'};

dictsizes = [256,512,1024];
ndictsizes = length(dictsizes);

%[ set parameters
patch_size = 5;
lambda = 0.1;                   % sparsity regularization
overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxIter = 10;                   % if 0, do not use backprojection

%[ other parameters
pruningVarThresh = 10;

for ss=3 %2:nsamplesizes
    samplesize = samplesizes(ss);
    for aa=3 %2:ndictsizes
        dictsize = dictsizes(aa);
        
        dictstring = sprintf('%dk',samplesize);
        dictDir = fullfile(dictroot,dictstring);

        outstring = sprintf('nsamples%d-dictsize%d',samplesize,dictsize);

%         D_256_lam-0.1_patchsz-5_zoom-3_Yang.mat
        dictName = sprintf('D_%d_lam-0.1_patchsz-5_zoom-3_Yang.mat',dictsize);

        dictFileName = fullfile(dictDir,dictName);
%         load(dictFileName,'Dh','Dl');

        DLtic = tic; DLcpu = cputime;

        [Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
            run_Dictionary_Training( indir, dictDir, dictsize, patch_size, ...
            lambda, samplesize*1000, up_scale, maxIter, pruningVarThresh );
    
        DLtimer.elaptime = toc(DLtic);
        DLtimer.cputime = cputime - DLcpu;

        ftimes = fullfile(dictDir,sprintf('%s_train-times.mat',outstring));
        save(ftimes,"DLtimer");






    end
end




