
% change this to your path to /matlab_src !!!!
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% Dictionary training w/ and w/out third order features
Demo_ThirdOrdFeat_DictionaryTraining;

% ScSR w/ and w/out third order features
Demo_ThirdOrdFeat_ScSR;

