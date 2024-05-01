% =========================================================================
% Simple demo codes for image super-resolution via sparse representation
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
% Edited by: RJ | 03-28-2024
%  - fixed image_list, ...
% =========================================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('qtfm');
addpath(pwd);  % assuming that `pwd` == /sparse-coding-super-resolution/matlab_src

% clear; clc; close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[ define dirs
testDataDir = 'Data/Testing';
dictDir = 'newdicts';
nsizes = [100,50,25];
dictPatchsize = 5;
dictLam = 0.1;
dsizes = [256,512,1024];

dictNames = {};
for nn=1:length(nsizes)
    nsamp=nsizes(nn);
    for dd=1:length(dsizes)
        dictSize=dsizes(dd);
    
        dictName = strcat('D_',num2str(dictSize),'_lam-',...
            num2str(dictLam),'_patchsz-',num2str(dictPatchsize),...
            '.mat');
    end
end


%[ define dictionary (via params)    (% dictName = 'D_2048_0.1_3.mat';)

dictSize = 2048;
dictName = strcat('D_',num2str(dictSize),'_',...
    num2str(dictLam),'_',num2str(dictPatchsize,'.mat'));
dictPath = fullfile(dictDir,dictName);
if ~exist(dictPath,'file')
    warning('NO DICTIONARY FOUND:\n %s\n',dictPath);
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[ set parameters
lambda      = 0.2;                   % sparsity regularization
overlap     = dictPatchsize - 1; %2; %4;                    % the more overlap the better (i.e., 4 for patch size 5x5)
up_scale    = 3; %2;                   % scaling factor, depending on the trained dictionary
maxIter     = 20;                   % if 0, do not use backprojection

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[ images to run ScSR on
image_list = {'Lena.png'};


for i = 1:size(image_list,1)
    fn_full = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
    if exist(fn_full,'file')
        continue;
    end
    % read test image
    testImageName = sprintf('Data/Testing/%s',image_list{i});
    im_l = imread(testImageName);



    % load dictionary
%     dictFileName = 'NewDictionary/D_512_0.15_5_s2.mat';
    % dictFileName = 'Dictionary/D_1024_0.15_5.mat';
    dictFileName = 'Dictionary_rj/D_2048_0.1_3.mat';
    load(dictFileName,'Dh','Dl');

    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(im_l);
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);

    % image super-resolution based on sparse representation
    [im_h_y_1] = ScSR(im_l_y, up_scale, Dh, Dl, lambda, overlap);
    [im_h_y_2] = ScSR(im_h_y_1, up_scale, Dh, Dl, lambda, overlap);
    [im_h_y] = backprojection(im_h_y_2, im_l_y, maxIter);

    % upscale the chrominance simply by "bicubic" 
    [nrow, ncol] = size(im_h_y);

    im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
    im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

    im_h_ycbcr = zeros([nrow, ncol, 3]);
    im_h_ycbcr(:, :, 1) = im_h_y;
    im_h_ycbcr(:, :, 2) = im_h_cb;
    im_h_ycbcr(:, :, 3) = im_h_cr;
    im_h = ycbcr2rgb(uint8(im_h_ycbcr));
    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');

    %save image
    fn_full_sr = fullfile(sprintf('Results-Testing/%s_res_sr-test.png',image_list{i}(1:end-4)));
    if exist(fn_full_sr,'file')
        fprintf(' sr file exists!\n%s\n',fn_full_sr);
    end
    fid = fopen(fn_full_sr,'w+');
    fclose(fid);
    imwrite(im_h,fn_full_sr);

    fmat = sprintf('Results-Testing/%s_results_SR.mat',image_list{i}(1:end-4));
    save(fmat,'im_h','im_b','im_l','im_h_y_1','im_h_y_2','im_h_y','im_h_cb',...
        'im_h_cr','dictFileName','lambda','overlap','up_scale','maxIter','testImageName');

    disp('');

    if 0 == 1
    
        f=figure('position',[48 463 1405 403]);
        subplot(131);
        imagesc(im_b); 
        set(gca,'colormap',gray);
        title('bilinear interp');
        axis image; axis off;
        subplot(132);
        imagesc(im_h);
        set(gca,'colormap',gray);
        title('super-res');
        axis image; axis off;
        subplot(133);
        imshowpair(im_b,im_h,'diff');
        title('difference');
    
        fout = fullfile(sprintf('Data/Testing/%s_res_sr-test-comparison.png',image_list{i}(1:end-4)));
        print(f,fout,'-dpng');
    end
end %while

if 1 == 1
    i = 1;

    % fn_full_res = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
    fn_full_gnd = fullfile(sprintf('Data/Testing/Lena_gnd.bmp'));
    im = imread(fn_full_gnd);
    % % read ground truth image
    % im = imread('Data/Testing/House_Of_Cards_2013_S02E01_0135_wanted.png');
    
    
    % load super res image
%     fn_full_sr = fullfile(sprintf('Data/Testing/%s_res_sr-test.png',image_list{i}(1:end-4)));
    fn_full_sr = fullfile(sprintf('Results-Testing/%s_res_sr-test.png',image_list{i}(1:end-4)));
    im_h = imread(fn_full_sr); %imwrite(im_h,fn_full_sr);
    
    % compute PSNR for the illuminance channel
    bb_rmse = compute_rmse(im, im_b);
    sp_rmse = compute_rmse(im, im_h);
    [qssim_sp,~] = qssim(im, im_h);
    [qssim_in,~] = qssim(im, im_b);
    
    im_gray = rgb2gray(im);
    im_h_gray = rgb2gray(im_h);
    im_b_gray = rgb2gray(im_b);
    [ssim_sp,~] = ssim_index(im_gray,im_h_gray);
    [ssim_in,~] = ssim_index(im_gray,im_b_gray);
    
    % fn_full = fullfile('Data/Testing/House_Of_Cards_2013_S02E01_0135_wanted_res.png');
    % fid = fopen(fn_full,'w+');
    % fclose(fid);
    % imwrite(im_h,fn_full);
    
    bb_psnr = 20*log10(255/bb_rmse);
    sp_psnr = 20*log10(255/sp_rmse);
    
    fprintf('\n');
    fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
    fprintf('PSNR for Sparse Representation Recovery: %f dB\n\n', sp_psnr);

    fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
    fprintf('RMSE for Sparse Representation Recovery: %f dB\n\n', sp_rmse);

    fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_in);
    fprintf('SSIM for Sparse Representation Recovery: %f dB\n\n', ssim_sp);

    fprintf('QSSIM for Bicubic Interpolation: %f dB\n', qssim_in);
    fprintf('QSSIM for Sparse Representation Recovery: %f dB\n\n', qssim_sp);
    
    
    % show the images
    figure, 
    subplot(121);
    imshow(im_h);
    title('Sparse Recovery');
    subplot(122);
    imshow(im_b);
    title('Bicubic Interpolation');

    fmat = sprintf('Results-Testing/%s_res_sr-metrics.mat',image_list{i}(1:end-4));
    save(fmat,'bb_rmse','sp_rmse','qssim_sp','qssim_in','ssim_in','ssim_sp','bb_psnr','sp_psnr')
end