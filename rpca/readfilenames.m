
%% addpath
clear all; close all;
addpath('bss_eval');
addpath('example');
addpath(genpath('inexact_alm_rpca'));

mydir  = pwd;
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end)-1);
path_in = [newdir, filesep, 'dataset', filesep, 'mono', filesep, '*.wav'];
Files=dir(path_in);
for k=1:length(Files)
   filename=Files(k).name;
   [wavinmix, fs] = audioread(filename);
   splitfile = split(filename,'.');
   filename = splitfile{1};
   parm.outname = ['example', filesep, 'output', filesep, filename];
   parm.lambda = 1;
   parm.nFFT = 1024;
   parm.windowsize = 1024;
   parm.masktype = 1; %1: binary mask, 2: no mask
   parm.gain = 1;
   parm.power = 1;
   parm.fs = fs;
   outputs = rpca_mask_execute(wavinmix, parm);
   fprintf('Output separation results are in %s\n', parm.outname)
   fprintf('%s_E is the sparse part and %s_A is the low rank part\n', ...
    parm.outname, parm.outname)
end
