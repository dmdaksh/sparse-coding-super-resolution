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
image_list = {'Lena.png','Child.png','Face.png'};
% i = 1;
for i=3 %2:length(image_list)

    if i==3
        up_scale = 3; 
    else
        up_scale = 2;
    end
    
    % read test image
    fn_full = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
    testImageName = sprintf('Data/Testing/%s',image_list{i});
    im_l = imread(testImageName);
    
    % read ground truth image
    fn_full_gnd = fullfile(sprintf('Data/Testing/%s_gnd.bmp',image_list{i}(1:end-4)));
    im = imread(fn_full_gnd);
    
    % change color space, work on illuminance only
    im_l_ycbcr = rgb2ycbcr(im_l);
    im_l_y = im_l_ycbcr(:, :, 1);
    im_l_cb = im_l_ycbcr(:, :, 2);
    im_l_cr = im_l_ycbcr(:, :, 3);
    
    %[add noise to LR image luminance channel
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
    
    %[[[ Regular 2nd order features - ScSR
    % image super-resolution based on sparse representation
    [im_h_y_1_d2a] = ScSR(im_l_y_noisy, up_scale, Dh2, Dl2, lambda, overlap);
    if i==3
        im_h_y_2_d2b=im_h_y_1_d2a;
    else
        [im_h_y_2_d2b] = ScSR(im_h_y_1_d2a, up_scale, Dh2, Dl2, lambda, overlap);
    end
    [im_h_y_d2] = backprojection(im_h_y_2_d2b, im_l_y_noisy, maxIter);
    
    %%%%%%%
    % Get bicubic + nearest interpolated images
    [nrow, ncol] = size(im_h_y_d2);
    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');
    im_b_noisy = imresize(im_l_noisy, [nrow, ncol], 'bicubic');
    % bicubic interpolation for reference
    im_l_near = imresize(im_l, [nrow, ncol], 'nearest');
    im_l_near_noisy = imresize(im_l_noisy, [nrow, ncol], 'nearest');
    %%%%%%%
    % upscale the chrominance simply by "bicubic"
    im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
    im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
    
    %[[[ Regular 2nd order features - with backproj step
    im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
    im_h_ycbcr_d2(:, :, 1) = im_h_y_d2;
    im_h_ycbcr_d2(:, :, 2) = im_h_cb;
    im_h_ycbcr_d2(:, :, 3) = im_h_cr;
    im_h_d2_wproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
    
    %save image
    srfilename2 = sprintf('%s_res_sr-test_d2_wproj.png',image_list{i}(1:end-4));
    fn_full_sr2 = fullfile(outdir,srfilename2);
    if exist(fn_full_sr2,'file')
        fprintf(' sr file exists!\n%s\n',fn_full_sr2);
    end
    fid = fopen(fn_full_sr2,'w+');
    fclose(fid);
    imwrite(im_h_d2_wproj,fn_full_sr2);
    %compute stats
    [ stats_d2 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2_wproj );
    %save results
    fmat2 = sprintf('%s/%s_results_SR_d2.mat',outdir,image_list{i}(1:end-4));
    save(fmat2,'im_h_d2_wproj','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b',...
        'im_h_y_d2','im_h_cb',...
        'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
        'maxIter','testImageName','stats_d2'); % , 'im_h_y_2',
    disp('');
    
    
    %[[[ Regular 2nd order features - skip backproj step
    im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
    im_h_ycbcr_d2(:, :, 1) = im_h_y_2_d2b;
    im_h_ycbcr_d2(:, :, 2) = im_h_cb;
    im_h_ycbcr_d2(:, :, 3) = im_h_cr;
    im_h_d2_noproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
    %compute stats
    [ stats_d2_noproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2_noproj );
    %save image
    srfilename2b = sprintf('%s_res_sr-test_d2_noproj.png',image_list{i}(1:end-4));
    fn_full_sr2b = fullfile(outdir,srfilename2b);
    if exist(fn_full_sr2b,'file')
    fprintf(' sr file exists!\n%s\n',fn_full_sr2b);
    end
    fid = fopen(fn_full_sr2b,'w+');
    fclose(fid);
    imwrite(im_h_d2_noproj,fn_full_sr2b);
    %save results
    fmat2 = sprintf('%s/%s_results_SR_d2_noproj.mat',outdir,image_list{i}(1:end-4));
    save(fmat2,'im_h_d2_noproj','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b','im_h_y_d2','im_h_cb',...
    'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
    'maxIter','testImageName','stats_d2_noproj'); % , 'im_h_y_2',
    disp('');
    
    
    
    
    %% ScSR - w 3rd order features
    
    % image super-resolution based on sparse representation
    [im_h_y_1_d3a] = ScSR(im_l_y_noisy, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
    if i==3
        im_h_y_2_d3b=im_h_y_1_d3a;
    else
        [im_h_y_2_d3b] = ScSR(im_h_y_1_d3a, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
    end
    [im_h_y_d3] = backprojection(im_h_y_2_d3b, im_l_y_noisy, maxIter);
    
    %[[[ Smooth 2nd order features - with backproj step
    im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
    im_h_ycbcr_d3(:, :, 1) = im_h_y_d3;
    im_h_ycbcr_d3(:, :, 2) = im_h_cb;
    im_h_ycbcr_d3(:, :, 3) = im_h_cr;
    im_h_d3_wproj = ycbcr2rgb(uint8(im_h_ycbcr_d3));
    
    %save image
    srfilename3 = sprintf('%s_res_sr-test_d3_wproj.png',image_list{i}(1:end-4));
    fn_full_sr3 = fullfile(outdir,srfilename3);
    if exist(fn_full_sr3,'file')
    fprintf(' sr file exists!\n%s\n',fn_full_sr3);
    end
    fid = fopen(fn_full_sr3,'w+');
    fclose(fid);
    imwrite(im_h_d3_wproj,fn_full_sr3);
    %compute stats
    [ stats_d3 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3_wproj );
    %save results
    fmat3 = sprintf('%s/%s_results_SR_d3_wproj.mat',outdir,image_list{i}(1:end-4));
    save(fmat3,'im_h_d3_wproj','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
    'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
    'maxIter','testImageName','stats_d3'); % , 'im_h_y_2',
    disp('');
    
    
    %[[[ Smooth 2nd order features - no backproj step
    im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
    im_h_ycbcr_d3(:, :, 1) = im_h_y_2_d3b;
    im_h_ycbcr_d3(:, :, 2) = im_h_cb;
    im_h_ycbcr_d3(:, :, 3) = im_h_cr;
    im_h_d3_noproj = ycbcr2rgb(uint8(im_h_ycbcr_d3));
    
    %save image
    srfilename3b = sprintf('%s_res_sr-test_d3_noproj.png',image_list{i}(1:end-4));
    fn_full_sr3b = fullfile(outdir,srfilename3b);
    if exist(fn_full_sr3b,'file')
    fprintf(' sr file exists!\n%s\n',fn_full_sr3b);
    end
    fid = fopen(fn_full_sr3b,'w+');
    fclose(fid);
    imwrite(im_h_d3_noproj,fn_full_sr3b);
    %compute stats
    [ stats_d3_noproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3_noproj );
    %save results
    fmat3 = sprintf('%s/%s_results_SR_d3_noproj.mat',outdir,image_list{i}(1:end-4));
    save(fmat3,'im_h_d3_noproj','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
    'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
    'maxIter','testImageName','stats_d3_noproj'); % , 'im_h_y_2',
    disp('');
    
    
    
    
    
    %% Compare results from {ord2, sm2}, {wproj, noproj}
    
    % display stats
    fprintf('\n');
    fprintf('PSNR for Bicubic:          %f dB\n', stats_d2.psnr.bi); %bb_psnr);
    fprintf('PSNR for ScSR, w proj:     %f dB\n', stats_d2.psnr.sr); %
    fprintf('PSNR for SmScSR, w proj:   %f dB\n', stats_d3.psnr.sr); %
    fprintf('PSNR for ScSR, no proj:    %f dB\n', stats_d2_noproj.psnr.sr); %
    fprintf('PSNR for SmScSR, no proj:  %f dB\n\n', stats_d3_noproj.psnr.sr); %
    
    fprintf('RMSE for Bicubic:          %f \n', stats_d2.rmse.bi);
    fprintf('RMSE for ScSR, w proj:     %f \n', stats_d2.rmse.sr);
    fprintf('RMSE for SmScSR, w proj:   %f \n', stats_d3.rmse.sr);
    fprintf('RMSE for ScSR, no proj:    %f \n', stats_d2_noproj.rmse.sr);
    fprintf('RMSE for SmScSR, no proj:  %f \n\n', stats_d3_noproj.rmse.sr);
    
    fprintf('SSIM for Bicubic:          %f \n', stats_d2.ssim.bi);
    fprintf('SSIM for ScSR, w proj:     %f \n', stats_d2.ssim.sr);
    fprintf('SSIM for SmScSR, w proj:   %f \n', stats_d3.ssim.sr);
    fprintf('SSIM for ScSR, no proj:    %f \n', stats_d2_noproj.ssim.sr);
    fprintf('SSIM for SmScSR, no proj:  %f \n\n', stats_d3_noproj.ssim.sr);
    
    fprintf('QSSIM for Bicubic:         %f \n', stats_d2.qssim.bi);
    fprintf('QSSIM for ScSR, w proj:    %f \n', stats_d2.qssim.sr);
    fprintf('QSSIM for SmScSR, w proj:  %f \n', stats_d3.qssim.sr);
    fprintf('QSSIM for ScSR, no proj:   %f \n', stats_d2_noproj.qssim.sr);
    fprintf('QSSIM for SmScSR, no proj: %f \n\n', stats_d3_noproj.qssim.sr);
    
    
    %% Compare w + wout smoothing 2nd order feats, w/backproj
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
        imagesc(im_h_d2_wproj);
        set(gca,'colormap',gray);
        title('SR');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(247);
        imshowpair(im,im_h_d2_wproj,'diff');
        title('Diff(SR vs. GT)');
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
        imagesc(im_h_d3_wproj);
        set(gca,'colormap',gray);
        title('SR-sm');
        set(gca,'FontSize',20);
        axis image; axis off;
        subplot(248);
        imshowpair(im,im_h_d3_wproj,'diff');
        title('Diff(SR-sm vs. GT)');
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
        
        fplotname = sprintf('%s_res_sr-test-comparison_wproj_noSm2-vs-Sm2.png',image_list{i}(1:end-4));
        fout = fullfile(outdir,fplotname);
        print(f,fout,'-dpng');
    end
    
    
    %% Compare w + wout smoothing 2nd order feats, w/out backproj
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
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2_noproj.rmse.bi);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2_noproj.psnr.bi);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2_noproj.ssim.bi);
        an1=annotation(gcf,'textbox',...
        [0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        subplot(243);
        imagesc(im_h_d2_noproj);
        set(gca,'colormap',gray);
        title('SR-NoProj');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(247);
        imshowpair(im,im_h_d2_noproj,'diff');
        title('Diff(SR-NoProj vs. GT)');
        set(gca,'FontSize',20);
        ax=gca; ax.Position(2) = 0.04;
        
        % Create textbox
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2_noproj.rmse.sr);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2_noproj.psnr.sr);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2_noproj.ssim.sr);
        an2=annotation(gcf,'textbox',...
        [0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        subplot(244);
        imagesc(im_h_d3_noproj);
        set(gca,'colormap',gray);
        title('SR-sm-NoProj');
        set(gca,'FontSize',20);
        axis image; axis off;
        subplot(248);
        imshowpair(im,im_h_d3_noproj,'diff');
        title('Diff(SR-sm-NoProj vs. GT)');
        set(gca,'FontSize',20);
        ax=gca; ax.Position(2) = 0.04;
        
        % Create textbox
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3_noproj.rmse.sr);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3_noproj.psnr.sr);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3_noproj.ssim.sr);
        an3=annotation(gcf,'textbox',...
        [0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        fplotname = sprintf('%s_res_sr-test-comparison_noproj_noSm2-vs-Sm2.png',image_list{i}(1:end-4));
        fout = fullfile(outdir,fplotname);
        print(f,fout,'-dpng');
    end
    
    
    
    
    
    
    
    %% Compare - zoom
    
    if i<3
        yzoominds = (213:333);
        xzoominds = (200:320);
    else
        yzoominds = (125:225);
        xzoominds = (65:165);
    end
    
    %% Compare w + wout smoothing 2nd order feats, w/backproj
    if 1 == 1
        f=figure('position',[50 53 1431 716],'color','w');
        
        subplot(241);
        imagesc(im(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Ground truth HR');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(245);
        imagesc(im_l_near_noisy(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Input Noisy LR');
        set(gca,'FontSize',20);
        axis image; axis off;
        ax=gca; ax.Position(2) = 0.04;
        
        subplot(242);
        imagesc(im_b_noisy(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Bilinear');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(246);
        imshowpair(im(xzoominds,yzoominds),im_b_noisy(xzoominds,yzoominds),'diff');
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
        imagesc(im_h_d2_wproj(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('SR');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(247);
        imshowpair(im(xzoominds,yzoominds),im_h_d2_wproj(xzoominds,yzoominds),'diff');
        title('Diff(SR vs. GT)');
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
        imagesc(im_h_d3_wproj(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('SR-sm');
        set(gca,'FontSize',20);
        axis image; axis off;
        subplot(248);
        imshowpair(im(xzoominds,yzoominds),im_h_d3_wproj(xzoominds,yzoominds),'diff');
        title('Diff(SR-sm vs. GT)');
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
        
        fplotname = sprintf('%s_res_sr-test-comparison_wproj_noSm2-vs-Sm2-zoom.png',image_list{i}(1:end-4));
        fout = fullfile(outdir,fplotname);
        print(f,fout,'-dpng');
    end
    
    
    %% Compare w + wout smoothing 2nd order feats, w/out backproj
    if 1 == 1
        f=figure('position',[50 53 1431 716],'color','w');
        
        subplot(241);
        imagesc(im(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Ground truth HR');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(245);
        imagesc(im_l_near_noisy(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Input Noisy LR');
        set(gca,'FontSize',20);
        axis image; axis off;
        ax=gca; ax.Position(2) = 0.04;
        
        subplot(242);
        imagesc(im_b_noisy(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('Bilinear');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(246);
        imshowpair(im(xzoominds,yzoominds),im_b_noisy(xzoominds,yzoominds),'diff');
        set(gca,'colormap',gray);
        title('Diff(Bi. vs. GT)');
        set(gca,'FontSize',20);
        axis image; axis off;
        ax=gca; ax.Position(2) = 0.04;
        
        % Create textbox
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2_noproj.rmse.bi);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2_noproj.psnr.bi);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2_noproj.ssim.bi);
        an1=annotation(gcf,'textbox',...
        [0.3392250174703 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        subplot(243);
        imagesc(im_h_d2_noproj(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('SR-NoProj');
        set(gca,'FontSize',20);
        axis image; axis off;
        
        subplot(247);
        imshowpair(im(xzoominds,yzoominds),im_h_d2_noproj(xzoominds,yzoominds),'diff');
        title('Diff(SR-NoProj vs. GT)');
        set(gca,'FontSize',20);
        ax=gca; ax.Position(2) = 0.04;
        
        % Create textbox
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d2_noproj.rmse.sr);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d2_noproj.psnr.sr);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d2_noproj.ssim.sr);
        an2=annotation(gcf,'textbox',...
        [0.547470999301187 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        subplot(244);
        imagesc(im_h_d3_noproj(xzoominds,yzoominds));
        set(gca,'colormap',gray);
        title('SR-sm-NoProj');
        set(gca,'FontSize',20);
        axis image; axis off;
        subplot(248);
        imshowpair(im(xzoominds,yzoominds),im_h_d3_noproj(xzoominds,yzoominds),'diff');
        title('Diff(SR-sm-NoProj vs. GT)');
        set(gca,'FontSize',20);
        ax=gca; ax.Position(2) = 0.04;
        
        % Create textbox
        bstrings{1} = sprintf('\\textbf{RMSE}: %g',stats_d3_noproj.rmse.sr);
        bstrings{2} = sprintf('\\textbf{PSNR}: %g',stats_d3_noproj.psnr.sr);
        bstrings{3} = sprintf('\\textbf{SSIM}: %g',stats_d3_noproj.ssim.sr);
        an3=annotation(gcf,'textbox',...
        [0.754319357092941 0.434357541899441 0.152039832285115 0.146648044692737],...
        'String',bstrings,...
        'FitBoxToText','off',...
        'FontSize',25,'FontWeight','bold',...
        'LineStyle','none',...
        'Interpreter','latex');
        
        fplotname = sprintf('%s_res_sr-test-comparison_noproj_noSm2-vs-Sm2-zoom.png',image_list{i}(1:end-4));
        fout = fullfile(outdir,fplotname);
        print(f,fout,'-dpng');
    end

end
