
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_mod_feat3';
if ~exist(outdir,'dir'), mkdir(outdir); end

%% zoom 3, patch 3

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_mod';
if ~exist(outdir,'dir'), mkdir(outdir); end


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
run_Dictionary_Training_mod( indir, outdir, dict_size, patch_size, ...
    lambda, nSmp, upscaleFactor, numIters, pruningVarThresh );

DLtimer.elaptime = toc(DLtic);
DLtimer.cputime = cputime - DLcpu;

[foutdir,foutname,~] = fileparts(dlparams.dict_path);
prof_dir = [foutdir filesep 'DictProfiler-' foutname];

profinfo = profile('info');
profsave(profinfo,prof_dir);


%% zoom 2, patch 5 - w smoothed 2nd order derivative

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_mod_sm2';
if ~exist(outdir,'dir'), mkdir(outdir); end


%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 512;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 5;            % image patch size
patch_overlap = patch_size - 1;
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 2;      % upscaling factor
numIters = 10;


%[ other parameters
pruningVarThresh = 10;

profile on;

DLtic = tic; DLcpu = cputime;

[Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
run_Dictionary_Training_mod2( indir, outdir, dict_size, patch_size, ...
    lambda, nSmp, upscaleFactor, numIters, pruningVarThresh );

DLtimer.elaptime = toc(DLtic);
DLtimer.cputime = cputime - DLcpu;

[foutdir,foutname,~] = fileparts(dlparams.dict_path);
prof_dir = [foutdir filesep 'DictProfiler-' foutname];

profinfo = profile('info');
profsave(profinfo,prof_dir);




%% zoom 2, patch 5 - w 3rd order LR feat

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_mod_feat3';
if ~exist(outdir,'dir'), mkdir(outdir); end


%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 512;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 5;            % image patch size
patch_overlap = patch_size - 1;
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 2;      % upscaling factor
numIters = 10;


%[ other parameters
pruningVarThresh = 10;

profile on;

DLtic = tic; DLcpu = cputime;

[Dh, Dl, dict_timers,dlparams, S, sparsecode_stat, Xh, Xl ] = ...
run_Dictionary_Training_mod( indir, outdir, dict_size, patch_size, ...
    lambda, nSmp, upscaleFactor, numIters, pruningVarThresh );

DLtimer.elaptime = toc(DLtic);
DLtimer.cputime = cputime - DLcpu;

[foutdir,foutname,~] = fileparts(dlparams.dict_path);
prof_dir = [foutdir filesep 'DictProfiler-' foutname];

profinfo = profile('info');
profsave(profinfo,prof_dir);



%% zoom 2, patch 5  - w/out 3rd order LR feat

%[ location of trianing images
indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
% indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/data/train_hr';

%[ location to save dictionary
outdir = 'Dictionary_rj_mod_no3';
if ~exist(outdir,'dir'), mkdir(outdir); end


%[ dictionary parameters (run useAuthorsDictParams.m to get authors params)
dict_size   = 512;          % dictionary size
lambda      = 0.15;         % sparsity regularization
patch_size  = 5;            % image patch size
patch_overlap = patch_size - 1;
nSmp        = 100000;       % number of patches to sample
upscaleFactor     = 2;      % upscaling factor
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





