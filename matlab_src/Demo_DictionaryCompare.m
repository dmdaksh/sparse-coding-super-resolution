%% Script to run dictionary comparison
%  - performs dictionary training using different parameters:
%   - # of dictionary atoms = 256, 512, 1024
%   - # of randomly sampled image patches = 25000, 50000, 100000
% 
% Calls: 
%   submit_DictionaryTraining_comparison.m [runs dictionary training]
%   Demo_SR_DictCompare.m [runs ScSR restoration, computes IQ metrics]
%
% RJ | 04-2024 | EECS 556 W24 Project | Group 8

% Change the path below to the location of your /matlab_src directory!!!
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

% Adds necessary paths
addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% path to training images
indir = 'Data/Training';
    
% path to save dictionaries
dictroot = 'newdicts';
if ~exist(dictroot,'dir'), mkdir(dictroot); end

% path to save comparison results
outdir = 'New-Results-dict-compare';
if ~exist(outdir,'dir'), mkdir(outdir); end


% run dictionary training 
DictFiles = submit_DictionaryTraining_comparison( indir, dictroot );

% run ScSR, get image quality metrics + compare
ComparisonResults = Demo_SR_DictCompare( dictroot, outdir );

