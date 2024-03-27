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

addpath(genpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/sparseCodingSuperResolution-master'));
addpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/qtfm');

clear all; clc;

image_list = {'Lena.png'};

%     {'The_Big_Bang_Theory1_S19E01_0248_wanted.png';
%     'The_Simpsons_S19E01_0003_wanted.png'
%     };
for i = 1:size(image_list,1)
    fn_full = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
%     if exist(fn_full,'file')
%         continue;
%     end
    % read test image
    im_l = imread(sprintf('Data/Testing/%s',image_list{i}));

    % set parameters
    lambda = 0.2;                   % sparsity regularization
    overlap = 4;                    % the more overlap the better (patch size 5x5)
    up_scale = 2;                   % scaling factor, depending on the trained dictionary
    maxIter = 20;                   % if 0, do not use backprojection

    % load dictionary
    load('Dictionary/D_1024_0.15_5.mat');

    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(im_l);
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);

    % image super-resolution based on sparse representation
    [im_h_y_1] = ScSR(im_l_y, 2, Dh, Dl, lambda, overlap);
    [im_h_y_2] = ScSR(im_h_y_1, 2, Dh, Dl, lambda, overlap);
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
    fn_full_sr = fullfile(sprintf('Data/Testing/%s_res_sr-test.png',image_list{i}(1:end-4)));
    if exist(fn_full_sr,'file')
        fprintf(' sr file exists!\n%s\n',fn_full_sr);
    end
    fid = fopen(fn_full_sr,'w+');
    fclose(fid);
    imwrite(im_h,fn_full_sr);

    disp('');

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
end %while

% fn_full_res = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
fn_full_gnd = fullfile(sprintf('Data/Testing/Lena_gnd.bmp'));
im_gt = imread(fn_full_gnd);
% % read ground truth image
% im = imread('Data/Testing/House_Of_Cards_2013_S02E01_0135_wanted.png');


% load super res image
fn_full_sr = fullfile(sprintf('Data/Testing/%s_res_sr-test.png',image_list{i}(1:end-4)));
im_h = imread(fn_full_sr); %imwrite(im_h,fn_full_sr);

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im_gt, im_b);
sp_rmse = compute_rmse(im_gt, im_h);
[qssim_sp,~] = qssim(im_gt, im_h);
[qssim_in,~] = qssim(im_gt, im_b);

im_gray = rgb2gray(im_gt);
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


fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);

disp([num2str(bb_psnr),num2str(ssim_in),num2str(qssim_in)]);

disp([num2str(sp_psnr),num2str(ssim_sp),num2str(qssim_sp)]);
% show the images
figure, imshow(im_h);
title('Sparse Recovery');
figure, imshow(im_b);
title('Bicubic Interpolation');