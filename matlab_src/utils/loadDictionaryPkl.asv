function D = loadDictionaryPkl(pkl_file)

% addpath('utils'); %/matlab_src/utils, contains py2mat.m
pickle = py.importlib.import_module('pickle');

%[load Dl pkl, convert to mat
fh = py.open(pkl_file, 'rb');
P = pickle.load(fh);    % pickle file loaded to Python variable
fh.close();
D = py2mat(P);         % pickle data converted to MATLAB native variable
