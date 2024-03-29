
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_2';

%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 1024;          % dictionary size
lambda      = 0.1;         % sparsity regularization
patch_size  = 3;            % image patch size
patch_overlap = patch_size - 1;
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 3;            % upscaling factor
numIters = 10;

%[ other parameters
pruningVarThresh = 10;

profile on;

DLtic = tic; DLcpu = cputime;

[Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
run_Dictionary_Training( indir, outdir, dict_size, patch_size, ...
    lambda, nSmp, upscaleFactor, numIters, pruningVarThresh );

DLtimer.elaptime = toc(DLtic);
DLtimer.cputime = cputime - DLcpu;

[foutdir,foutname,~] = fileparts(dlparams.dict_path);
prof_dir = [foutdir filesep 'DictProfiler-' foutname];

profinfo = profile('info');
profsave(profinfo,prof_dir);


