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

cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% addpath(genpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/sparseCodingSuperResolution-master'));
% addpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/qtfm');

clear Dl2 Dl3 Dh2 Dh3

%% with 3rd order LR features

outdir = 'New-Results-mod-sm2-comparison-correct+noise';
if ~exist(outdir,'dir'), mkdir(outdir); end

dictDir2 = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Dictionary_rj_mod_no3';
dictName2 = 'D_512_lam-0.15_patchsz-5_zoom-2_Yang.mat';
dictFileName2 = fullfile(dictDir2,dictName2);
load(dictFileName2,'Dh','Dl');
Dh2 = Dh;
Dl2 = Dl;

dictDir3 = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Dictionary_rj_mod_sm2';
dictName3 = 'D_512_lam-0.15_patchsz-5_zoom-2_Yang.mat';
dictFileName3 = fullfile(dictDir3,dictName3);
load(dictFileName3,'Dh','Dl');
Dh3 = Dh;
Dl3 = Dl;

%[ set parameters
upscaleFactor = 2;
patch_size = 5;
lambda = 0.15;                   % sparsity regularization
overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
up_scale = 2;                   % scaling factor, depending on the trained dictionary
maxIter = 10;                   % if 0, do not use backprojection

%[ load image
image_list = {'Lena.png'};
i = 1;

% read test image
fn_full = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
testImageName = sprintf('Data/Testing/%s',image_list{i});
im_l = imread(testImageName);

% read ground truth image
fn_full_gnd = fullfile(sprintf('Data/Testing/Lena_gnd.bmp'));
im = imread(fn_full_gnd);

% change color space, work on illuminance only
im_l_ycbcr = rgb2ycbcr(im_l);
im_l_y = im_l_ycbcr(:, :, 1);
im_l_cb = im_l_ycbcr(:, :, 2);
im_l_cr = im_l_ycbcr(:, :, 3);


nvariance = 5e-4;
im_l_y_noisy = imnoise(im_l_y,"gaussian",0,nvariance);

figure('color','w','position',[740 640 371 158]);
subplot(121); imshow(im_l_y); title('orig LR Y-channel');
subplot(122); imshow(im_l_y_noisy); title(sprintf('noisy, awgn var=%g',nvariance));
print(gcf,fullfile(outdir,'noisy_lr_image_Y-channel.png'),'-dpng','-r300');

tmpnoisy = zeros(size(im_l),'uint8');
tmpnoisy(:,:,1) = im_l_y_noisy;
tmpnoisy(:,:,2) = im_l_cb;
tmpnoisy(:,:,3) = im_l_cr;
im_l_noisy = ycbcr2rgb(tmpnoisy);

figure('color','w','position',[740 640 371 158]);
subplot(121); imshow(im_l); title('orig LR');
subplot(122); imshow(im_l_noisy); title(sprintf('noisy, awgn var=%g',nvariance));
print(gcf,fullfile(outdir,'noisy_lr_image_rgb.png'),'-dpng','-r300');


%% ScSR - no 3rd order features

%% ScSR - no 3rd order features
% image super-resolution based on sparse representation
[im_h_y_1_d2a] = ScSR(im_l_y_noisy, up_scale, Dh2, Dl2, lambda, overlap);
[im_h_y_2_d2b] = ScSR(im_h_y_1_d2a, up_scale, Dh2, Dl2, lambda, overlap);
[im_h_y_d2] = backprojection(im_h_y_2_d2b, im_l_y_noisy, maxIter);

% upscale the chrominance simply by "bicubic"
[nrow, ncol] = size(im_h_y_d2);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
im_h_ycbcr_d2(:, :, 1) = im_h_y_d2;
im_h_ycbcr_d2(:, :, 2) = im_h_cb;
im_h_ycbcr_d2(:, :, 3) = im_h_cr;
im_h_d2 = ycbcr2rgb(uint8(im_h_ycbcr_d2));

% bicubic interpolation for reference
im_b = imresize(im_l, [nrow, ncol], 'bicubic');
im_b_noisy = imresize(im_l_noisy, [nrow, ncol], 'bicubic');

% bicubic interpolation for reference
im_l_near = imresize(im_l, [nrow, ncol], 'nearest');
im_l_near_noisy = imresize(im_l_noisy, [nrow, ncol], 'nearest');

%save image
srfilename2 = sprintf('%s_res_sr-test_d2.png',image_list{i}(1:end-4));
fn_full_sr2 = fullfile(outdir,srfilename2);
if exist(fn_full_sr2,'file')
    fprintf(' sr file exists!\n%s\n',fn_full_sr2);
end
fid = fopen(fn_full_sr2,'w+');
fclose(fid);
imwrite(im_h_d2,fn_full_sr2);

[ stats_d2 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2 );

fmat2 = sprintf('%s/%s_results_SR_d2.mat',outdir,image_list{i}(1:end-4));
save(fmat2,'im_h_d2','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b','im_h_y_d2','im_h_cb',...
'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
'maxIter','testImageName','stats_d2'); % , 'im_h_y_2',
disp('');

if 1 == 1
    f=figure('position',[50 53 1431 716]);
    subplot(231);
    imagesc(im);
    set(gca,'colormap',gray);
    title('Ground truth HR');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(234);
    imagesc(im_l_noisy);
    set(gca,'colormap',gray);
    title('Input LR');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(232);
    imagesc(im_b_noisy);
    set(gca,'colormap',gray);
    title('Bilinear');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(235);
    imshowpair(im,im_b_noisy,'diff');
    set(gca,'colormap',gray);
    title('Diff(Bi. vs. GT)');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(233);
    imagesc(im_h_d2);
    set(gca,'colormap',gray);
    title('SR');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(236);
    imshowpair(im,im_h_d2,'diff');
    title('Diff(SR vs. GT)');
    set(gca,'FontSize',20);
    fplotname = sprintf('%s_res_sr-test-comparison_d2.png',image_list{i}(1:end-4));
    fout = fullfile(outdir,fplotname);
    print(f,fout,'-dpng');
end


%% ScSR - w 3rd order features

% image super-resolution based on sparse representation
[im_h_y_1_d3a] = ScSR(im_l_y_noisy, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
[im_h_y_2_d3b] = ScSR(im_h_y_1_d3a, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
[im_h_y_d3] = backprojection(im_h_y_2_d3b, im_l_y_noisy, maxIter);

% upscale the chrominance simply by "bicubic"
[nrow, ncol] = size(im_h_y_d3);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
im_h_ycbcr_d3(:, :, 1) = im_h_y_d3;
im_h_ycbcr_d3(:, :, 2) = im_h_cb;
im_h_ycbcr_d3(:, :, 3) = im_h_cr;
im_h_d3 = ycbcr2rgb(uint8(im_h_ycbcr_d3));

%save image
srfilename3 = sprintf('%s_res_sr-test_d3.png',image_list{i}(1:end-4));
fn_full_sr3 = fullfile(outdir,srfilename3);
if exist(fn_full_sr3,'file')
fprintf(' sr file exists!\n%s\n',fn_full_sr3);
end
fid = fopen(fn_full_sr3,'w+');
fclose(fid);
imwrite(im_h_d2,fn_full_sr3);

[ stats_d3 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3 );

fmat3 = sprintf('%s/%s_results_SR_d3.mat',outdir,image_list{i}(1:end-4));
save(fmat3,'im_h_d3','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
'maxIter','testImageName','stats_d3'); % , 'im_h_y_2',
disp('');

% display stats
fprintf('\n');
fprintf('PSNR for Bicubic: %f dB\n', stats_d2.psnr.bi); %bb_psnr);
fprintf('PSNR for ScSR:    %f dB\n', stats_d2.psnr.sr); %
fprintf('PSNR for SmScSR:  %f dB\n\n', stats_d3.psnr.sr); %

fprintf('RMSE for Bicubic: %f \n', stats_d2.rmse.bi);
fprintf('RMSE for ScSR:    %f \n', stats_d2.rmse.sr);
fprintf('RMSE for SmScSR:  %f \n\n', stats_d3.rmse.sr);

fprintf('SSIM for Bicubic: %f \n', stats_d2.ssim.bi);
fprintf('SSIM for ScSR:    %f \n', stats_d2.ssim.sr);
fprintf('SSIM for SmScSR:  %f \n\n', stats_d3.ssim.sr);

fprintf('QSSIM for Bicubic: %f \n', stats_d2.qssim.bi);
fprintf('QSSIM for ScSR:    %f \n', stats_d2.qssim.sr);
fprintf('QSSIM for SmScSR:  %f \n\n', stats_d3.qssim.sr);

if 1 == 1
    f=figure('position',[50 53 1431 716]);
    
    subplot(231);
    imagesc(im);
    set(gca,'colormap',gray);
    title('Ground truth HR');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(234);
    imagesc(im_l_noisy);
    set(gca,'colormap',gray);
    title('Input LR');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(232);
    imagesc(im_b_noisy);
    set(gca,'colormap',gray);
    title('Bilinear');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(235);
    imshowpair(im,im_b_noisy,'diff');
    set(gca,'colormap',gray);
    title('Diff(Bi. vs. GT)');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(233);
    imagesc(im_h_d3);
    set(gca,'colormap',gray);
    title('SR');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(236);
    imshowpair(im,im_h_d3,'diff');
    title('Diff(SR vs. GT)');
    set(gca,'FontSize',20);
    
    fplotname = sprintf('%s_res_sr-test-comparison_d3.png',image_list{i}(1:end-4));
    fout = fullfile(outdir,fplotname);
    print(f,fout,'-dpng');
end


%% Compare w + wout 3rd order LR features
if 1 == 1
    f=figure('position',[50 53 1431 716],'color','w');
    
    subplot(241);
    imagesc(im);
    set(gca,'colormap',gray);
    title('Ground truth HR');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(245);
    imagesc(im_l_noisy);
    set(gca,'colormap',gray);
    title('Input Noisy LR');
    set(gca,'FontSize',20);
    axis image; axis off;
    ax=gca; ax.Position(2) = 0.04;
    
    subplot(242);
    imagesc(im_b_noisy);
    set(gca,'colormap',gray);
    title('Bilinear');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(246);
    imshowpair(im,im_b_noisy,'diff');
    set(gca,'colormap',gray);
    title('Diff(Bi. vs. GT)');
    set(gca,'FontSize',20);
    axis image; axis off;
    ax=gca; ax.Position(2) = 0.04;
    
    % Create textbox
    bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.bi);
    bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.bi);
    bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.bi);
    an1=annotation(gcf,'textbox',...
    [0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
    'String',bstrings,...
    'FitBoxToText','off',...
    'FontSize',25,'FontWeight','bold',...
    'LineStyle','none',...
    'Interpreter','latex');
    
    subplot(243);
    imagesc(im_h_d2);
    set(gca,'colormap',gray);
    title('SR-ord2');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(247);
    imshowpair(im,im_h_d2,'diff');
    title('Diff(SR-ord2 vs. GT)');
    set(gca,'FontSize',20);
    ax=gca; ax.Position(2) = 0.04;
    
    % Create textbox
    bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.sr);
    bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.sr);
    bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.sr);
    an2=annotation(gcf,'textbox',...
    [0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
    'String',bstrings,...
    'FitBoxToText','off',...
    'FontSize',25,'FontWeight','bold',...
    'LineStyle','none',...
    'Interpreter','latex');
    
    subplot(244);
    imagesc(im_h_d3);
    set(gca,'colormap',gray);
    title('SR-ord3');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(248);
    imshowpair(im,im_h_d3,'diff');
    title('Diff(SR-ord3 vs. GT)');
    set(gca,'FontSize',20);
    ax=gca; ax.Position(2) = 0.04;
    
    % Create textbox
    bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3.rmse.sr);
    bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3.psnr.sr);
    bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3.ssim.sr);
    an3=annotation(gcf,'textbox',...
    [0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
    'String',bstrings,...
    'FitBoxToText','off',...
    'FontSize',25,'FontWeight','bold',...
    'LineStyle','none',...
    'Interpreter','latex');
    
    fplotname = sprintf('%s_res_sr-test-comparison_d2+d3.png',image_list{i}(1:end-4));
    fout = fullfile(outdir,fplotname);
    print(f,fout,'-dpng');
end



%% Compare w + wout 3rd order LR features
if 1 == 1
    f=figure('position',[50 53 1431 716],'color','w');
    
    subplot(242);
    imagesc(im);
    set(gca,'colormap',gray);
    title('Ground truth HR');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(246);
    imagesc(im_l_noisy);
    set(gca,'colormap',gray);
    title('Input Noisy LR');
    set(gca,'FontSize',20);
    axis image; axis off;
    ax=gca; ax.Position(2) = 0.04;
    
%     subplot(242);
%     imagesc(im_b_noisy);
%     set(gca,'colormap',gray);
%     title('Bilinear');
%     set(gca,'FontSize',20);
%     axis image; axis off;
%     
%     subplot(246);
%     imshowpair(im,im_b_noisy,'diff');
%     set(gca,'colormap',gray);
%     title('Diff(Bi. vs. GT)');
%     set(gca,'FontSize',20);
%     axis image; axis off;
%     ax=gca; ax.Position(2) = 0.04;
%     
%     % Create textbox
%     bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.bi);
%     bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.bi);
%     bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.bi);
%     an1=annotation(gcf,'textbox',...
%     [0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
%     'String',bstrings,...
%     'FitBoxToText','off',...
%     'FontSize',25,'FontWeight','bold',...
%     'LineStyle','none',...
%     'Interpreter','latex');
    
    subplot(243);
    imagesc(im_h_d2);
    set(gca,'colormap',gray);
    title('SR-ord2');
    set(gca,'FontSize',20);
    axis image; axis off;
    
    subplot(247);
    imshowpair(im,im_h_d2,'diff');
    title('Diff(SR-ord2 vs. GT)');
    set(gca,'FontSize',20);
    ax=gca; ax.Position(2) = 0.04;
    
    % Create textbox
    bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.sr);
    bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.sr);
    bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.sr);
    an2=annotation(gcf,'textbox',...
    [0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
    'String',bstrings,...
    'FitBoxToText','off',...
    'FontSize',25,'FontWeight','bold',...
    'LineStyle','none',...
    'Interpreter','latex');
    
    subplot(244);
    imagesc(im_h_d3);
    set(gca,'colormap',gray);
    title('SR-ord3');
    set(gca,'FontSize',20);
    axis image; axis off;
    subplot(248);
    imshowpair(im,im_h_d3,'diff');
    title('Diff(SR-ord3 vs. GT)');
    set(gca,'FontSize',20);
    ax=gca; ax.Position(2) = 0.04;
    
    % Create textbox
    bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3.rmse.sr);
    bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3.psnr.sr);
    bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3.ssim.sr);
    an3=annotation(gcf,'textbox',...
    [0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
    'String',bstrings,...
    'FitBoxToText','off',...
    'FontSize',25,'FontWeight','bold',...
    'LineStyle','none',...
    'Interpreter','latex');
    
    fplotname = sprintf('%s_res_sr-test-comparison_d2+d3.png',image_list{i}(1:end-4));
    fout = fullfile(outdir,fplotname);
    print(f,fout,'-dpng');
end





%% Compare - zoom

yzoominds = (213:333);
xzoominds = (200:320);

f=figure('position',[50 53 1431 716],'color','w');

subplot(241);
imagesc(im(xzoominds,yzoominds,:));
set(gca,'colormap',gray);
title('Ground truth HR');
set(gca,'FontSize',20);
axis image; axis off;

subplot(245);
imagesc(im_l_near_noisy(xzoominds,yzoominds,:));
set(gca,'colormap',gray);
title('Input LR');
set(gca,'FontSize',20);
axis image; axis off;
ax=gca; ax.Position(2) = 0.04;

subplot(242);
imagesc(im_b_noisy(xzoominds,yzoominds,:));
set(gca,'colormap',gray);
title('Bilinear');
set(gca,'FontSize',20);
axis image; axis off;

subplot(246);
imshowpair(im(xzoominds,yzoominds,:),im_b_noisy(xzoominds,yzoominds,:),'diff');
set(gca,'colormap',gray);
title('Diff(Bi. vs. GT)');
set(gca,'FontSize',20);
axis image; axis off;
ax=gca; ax.Position(2) = 0.04;
% Create textbox
bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.bi);
bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.bi);
bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.bi);
an1=annotation(gcf,'textbox',...
[0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
'String',bstrings,...
'FitBoxToText','off',...
'FontSize',25,'FontWeight','bold',...
'LineStyle','none',...
'Interpreter','latex');

subplot(243);
imagesc(im_h_d2(xzoominds,yzoominds,:));
set(gca,'colormap',gray);
title('SR-ord2');
set(gca,'FontSize',20);
axis image; axis off;

subplot(247);
imshowpair(im(xzoominds,yzoominds,:),im_h_d2(xzoominds,yzoominds,:),'diff');
title('Diff(SR-ord2 vs. GT)');
set(gca,'FontSize',20);
ax=gca; ax.Position(2) = 0.04;
% Create textbox
bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.sr);
bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.sr);
bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.sr);
an2=annotation(gcf,'textbox',...
[0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
'String',bstrings,...
'FitBoxToText','off',...
'FontSize',25,'FontWeight','bold',...
'LineStyle','none',...
'Interpreter','latex');

subplot(244);
imagesc(im_h_d3(xzoominds,yzoominds,:));
set(gca,'colormap',gray);
title('SR-ord3');
set(gca,'FontSize',20);
axis image; axis off;
subplot(248);
imshowpair(im(xzoominds,yzoominds,:),im_h_d3(xzoominds,yzoominds,:),'diff');
title('Diff(SR-ord3 vs. GT)');
set(gca,'FontSize',20);
ax=gca; ax.Position(2) = 0.04;
% Create textbox
bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3.rmse.sr);
bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3.psnr.sr);
bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3.ssim.sr);
an3=annotation(gcf,'textbox',...
[0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
'String',bstrings,...
'FitBoxToText','off',...
'FontSize',25,'FontWeight','bold',...
'LineStyle','none',...
'Interpreter','latex');

fplotname = sprintf('%s_res_sr-test-comparison_d2+d3-zoom.png',image_list{i}(1:end-4));
fout = fullfile(outdir,fplotname);
print(f,fout,'-dpng');


%% Skip backproj step

% image super-resolution based on sparse representation
[im_h_y_1_d2a] = ScSR(im_l_y_noisy, up_scale, Dh2, Dl2, lambda, overlap);
[im_h_y_2_d2b] = ScSR(im_h_y_1_d2a, up_scale, Dh2, Dl2, lambda, overlap);

% upscale the chrominance simply by "bicubic"
[nrow, ncol] = size(im_h_y_2_d2b);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
im_h_ycbcr_d2(:, :, 1) = im_h_y_2_d2b;
im_h_ycbcr_d2(:, :, 2) = im_h_cb;
im_h_ycbcr_d2(:, :, 3) = im_h_cr;
im_h_d2 = ycbcr2rgb(uint8(im_h_ycbcr_d2));

% bicubic interpolation for reference
im_b = imresize(im_l, [nrow, ncol], 'bicubic');
im_b_noisy = imresize(im_l_noisy, [nrow, ncol], 'bicubic');

% bicubic interpolation for reference
im_l_near = imresize(im_l, [nrow, ncol], 'nearest');
im_l_near_noisy = imresize(im_l_noisy, [nrow, ncol], 'nearest');

%save image
srfilename2 = sprintf('%s_res_sr-test_d2_nobackproj.png',image_list{i}(1:end-4));
fn_full_sr2 = fullfile(outdir,srfilename2);
if exist(fn_full_sr2,'file')
fprintf(' sr file exists!\n%s\n',fn_full_sr2);
end
fid = fopen(fn_full_sr2,'w+');
fclose(fid);
imwrite(im_h_d2,fn_full_sr2);

[ stats_d2_nobackproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2 );

fmat2 = sprintf('%s/%s_results_SR_d2_nobackproj.mat',outdir,image_list{i}(1:end-4));
save(fmat2,'im_h_d2','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b','im_h_y_d2','im_h_cb',...
'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
'maxIter','testImageName','stats_d2_nobackproj'); % , 'im_h_y_2',
disp('');




% image super-resolution based on sparse representation
[im_h_y_1_d3a] = ScSR(im_l_y_noisy, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
[im_h_y_2_d3b] = ScSR(im_h_y_1_d3a, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
% [im_h_y_d3] = backprojection(im_h_y_2_d3b, im_l_y_noisy, maxIter);

% upscale the chrominance simply by "bicubic"
[nrow, ncol] = size(im_h_y_2_d3b);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
im_h_ycbcr_d3(:, :, 1) = im_h_y_2_d3b;
im_h_ycbcr_d3(:, :, 2) = im_h_cb;
im_h_ycbcr_d3(:, :, 3) = im_h_cr;
im_h_d3 = ycbcr2rgb(uint8(im_h_ycbcr_d3));

%save image
srfilename3 = sprintf('%s_res_sr-test_d3_nobackproj.png',image_list{i}(1:end-4));
fn_full_sr3 = fullfile(outdir,srfilename3);
if exist(fn_full_sr3,'file')
fprintf(' sr file exists!\n%s\n',fn_full_sr3);
end
fid = fopen(fn_full_sr3,'w+');
fclose(fid);
imwrite(im_h_d3,fn_full_sr3);

[ stats_d3_nobackproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3 );

fmat3 = sprintf('%s/%s_results_SR_d3_nobackproj.mat',outdir,image_list{i}(1:end-4));
save(fmat3,'im_h_d3','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
'maxIter','testImageName','stats_d3_nobackproj'); % , 'im_h_y_2',
disp('');

fprintf('\n');
fprintf('PSNR for Bicubic: %f dB\n', stats_d2_nobackproj.psnr.bi); %bb_psnr);
fprintf('PSNR for ScSR:    %f dB\n', stats_d2_nobackproj.psnr.sr); %
fprintf('PSNR for SmScSR:  %f dB\n\n', stats_d3_nobackproj.psnr.sr); %

fprintf('RMSE for Bicubic: %f \n', stats_d2_nobackproj.rmse.bi);
fprintf('RMSE for ScSR:    %f \n', stats_d2_nobackproj.rmse.sr);
fprintf('RMSE for SmScSR:  %f \n\n', stats_d3_nobackproj.rmse.sr);

fprintf('SSIM for Bicubic: %f \n', stats_d2_nobackproj.ssim.bi);
fprintf('SSIM for ScSR:    %f \n', stats_d2_nobackproj.ssim.sr);
fprintf('SSIM for SmScSR:  %f \n\n', stats_d3_nobackproj.ssim.sr);

fprintf('QSSIM for Bicubic: %f \n', stats_d2_nobackproj.qssim.bi);
fprintf('QSSIM for ScSR:    %f \n', stats_d2_nobackproj.qssim.sr);
fprintf('QSSIM for SmScSR:  %f \n\n', stats_d3_nobackproj.qssim.sr);




% 
% %% compare ord2 vs ord3
% f=figure('position',[50 53 1431 716],'color','w');
% subplot(241);
% imagesc(im(xzoominds,yzoominds,:));
% set(gca,'colormap',gray);
% title('Ground truth HR');
% set(gca,'FontSize',20);
% axis image; axis off;
% subplot(245);
% imagesc(im_l_near_noisy(xzoominds,yzoominds,:));
% set(gca,'colormap',gray);
% title('Input LR');
% set(gca,'FontSize',20);
% axis image; axis off;
% ax=gca; ax.Position(2) = 0.04;
% subplot(242);
% imagesc(im_b_noisy(xzoominds,yzoominds,:));
% set(gca,'colormap',gray);
% title('Bilinear');
% set(gca,'FontSize',20);
% axis image; axis off;
% subplot(246);
% imshowpair(im(xzoominds,yzoominds,:),im_b_noisy(xzoominds,yzoominds,:),'diff');
% set(gca,'colormap',gray);
% title('Diff(Bi. vs. GT)');
% set(gca,'FontSize',20);
% axis image; axis off;
% ax=gca; ax.Position(2) = 0.04;
% % Create textbox
% bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.bi);
% bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.bi);
% bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.bi);
% an1=annotation(gcf,'textbox',...
% [0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
% 'String',bstrings,...
% 'FitBoxToText','off',...
% 'FontSize',25,'FontWeight','bold',...
% 'LineStyle','none',...
% 'Interpreter','latex');
% subplot(243);
% imagesc(im_h_d2(xzoominds,yzoominds,:));
% set(gca,'colormap',gray);
% title('SR-ord2');
% set(gca,'FontSize',20);
% axis image; axis off;
% subplot(247);
% imshowpair(im(xzoominds,yzoominds,:),im_h_d2(xzoominds,yzoominds,:),'diff');
% title('Diff(SR-ord2 vs. GT)');
% set(gca,'FontSize',20);
% ax=gca; ax.Position(2) = 0.04;
% % Create textbox
% bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2.rmse.sr);
% bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2.psnr.sr);
% bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2.ssim.sr);
% an2=annotation(gcf,'textbox',...
% [0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
% 'String',bstrings,...
% 'FitBoxToText','off',...
% 'FontSize',25,'FontWeight','bold',...
% 'LineStyle','none',...
% 'Interpreter','latex');
% subplot(244);
% imagesc(im_h_d3(xzoominds,yzoominds,:));
% set(gca,'colormap',gray);
% title('SR-ord3');
% set(gca,'FontSize',20);
% axis image; axis off;
% subplot(248);
% imshowpair(im_h_d2(xzoominds,yzoominds,:),im_h_d3(xzoominds,yzoominds,:),'diff');
% title('Diff(SR-ord3 vs. SR-ord2)');
% set(gca,'FontSize',20);
% ax=gca; ax.Position(2) = 0.04;
% % Create textbox
% bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3.rmse.sr);
% bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3.psnr.sr);
% bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3.ssim.sr);
% an3=annotation(gcf,'textbox',...
% [0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
% 'String',bstrings,...
% 'FitBoxToText','off',...
% 'FontSize',25,'FontWeight','bold',...
% 'LineStyle','none',...
% 'Interpreter','latex');
% fplotname = sprintf('%s_res_sr-test-comparison_d2+d3-donly.png',image_list{i}(1:end-4));
% fout = fullfile(outdir,fplotname);
% print(f,fout,'-dpng');

