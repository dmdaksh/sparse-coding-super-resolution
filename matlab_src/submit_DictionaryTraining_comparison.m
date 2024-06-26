function [ DictFiles ] = submit_DictionaryTraining_comparison( indir, outroot )
% [ DictFiles ] = submit_DictionaryTraining_comparison( indir, outroot )
% Inputs:
%   indir = path to training images
%   outroot = path to save dictionaries
% Output: 
%   DictFiles = cell struct with names of learned dictionaries
%
%Function to submit dictionary training for dictionary comparison
%  - performs dictionary training using different parameters:
%   - # of dictionary atoms = 256, 512, 1024
%   - # of randomly sampled image patches = 25000, 50000, 100000
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

if nargin==0
    % %[ location of trianing images
    indir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Data/Training';
    
    % %[ location to save dictionaries to
    outroot = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/newdicts';
    if ~exist(outroot,'dir'), mkdir(outroot); end
end

%[ Parameters to change for dictionary training
samplesizes = [25,50,100];                  % # of randomly sampled patches (thousands)
nsamplesizes = length(samplesizes);         % 
dictsizes = [256,512,1024];                 % # of dictionary atoms
ndictsizes = length(dictsizes); 

%[ Parameters to fix for all dictionaries
patch_size = 5;
lambda = 0.1;                   % sparsity regularization
overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxIter = 10;                   % if 0, do not use backprojection
pruningVarThresh = 10;          % for pruning randomly sampled patches

DictFiles = cell(nsamplesizes,ndictsizes);

% Loop through params and train dictionaries
for ss=1:nsamplesizes
    samplesize = samplesizes(ss);
    for aa=1:ndictsizes
        dictsize = dictsizes(aa);

        fprintf(' --Training dictionary with: %g atoms, %g random patches \n ',dictsize, samplesize*1e3);
        
        % subdirectory to save dict, based on # of random patches
        dictstring = sprintf('%dk',samplesize);
        dictDir = fullfile(dictroot,dictstring);
        if ~exist(dictDir,'dir'), mkdir(dictDir); end
        outstring = sprintf('nsamples%d-dictsize%d',samplesize,dictsize);
 
        % run dict training (and time how long it takes)
        DLtic = tic; DLcpu = cputime;
        [Dh, Dl, dict_timers, dlparams, S, sparsecode_stat, Xh, Xl ] = ...
            run_Dictionary_Training( indir, dictDir, dictsize, patch_size, ...
            lambda, samplesize*1000, up_scale, maxIter, pruningVarThresh );    
        DLtimer.elaptime = toc(DLtic);
        DLtimer.cputime = cputime - DLcpu;
        ftimes = fullfile(dictDir,sprintf('%s_train-times.mat',outstring));
        save(ftimes,"DLtimer");

        % the name of the output dicts (saved during run_Dictionary_Training.m)
        dictName = sprintf('D_%d_lam-0.1_patchsz-5_zoom-3_Yang.mat',dictsize);
        dictFileName = fullfile(dictDir,dictName);
        fprintf(' --Training complete - dicts saved to:\n %s\n',dictFileName);
        DictFiles{ss,aa} = dictFileName;

    end
end




