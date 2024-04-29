
cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');

% addpath(genpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/sparseCodingSuperResolution-master'));
% addpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/qtfm');


%% with 3rd order LR features

outdir = 'New-Results-dict-compare';
if ~exist(outdir,'dir'), mkdir(outdir); end

dictroot = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/newdicts';

samplesizes = [25,50,100];
nsamplesizes = length(samplesizes);

samplesizesString = {'25','50','100'};

dictsizes = [256,512,1024];
ndictsizes = length(dictsizes);

%[ set parameters
patch_size = 5;
lambda = 0.1;                   % sparsity regularization
overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxIter = 10;                   % if 0, do not use backprojection


allresults_bi = zeros(nsamplesizes,ndictsizes,4);
allresults_noproj = zeros(nsamplesizes,ndictsizes,4);
allresults_wproj = zeros(nsamplesizes,ndictsizes,4);
allfiles = cell(nsamplesizes,ndictsizes);

for ss=1:nsamplesizes
    samplesize = samplesizes(ss);
    for aa=1:ndictsizes
        dictsize = dictsizes(aa);
        
        dictstring = sprintf('%dk',samplesize);
        dictDir = fullfile(dictroot,dictstring);

        outstring = sprintf('nsamples%d-dictsize%d',samplesize,dictsize);

%         D_256_lam-0.1_patchsz-5_zoom-3_Yang.mat
        dictName = sprintf('D_%d_lam-0.1_patchsz-5_zoom-3_Yang.mat',dictsize);
        allfiles{ss,aa} = [dictstring filesep dictName];

        dictFileName = fullfile(dictDir,dictName);
        load(dictFileName,'Dh','Dl');

        %[ load image
        image_list = {'Face.png'};
        % i = 1;
        for i=1 %2:length(image_list)
            
            % read test image
            testImageName = sprintf('Data/Testing/%s',image_list{i});
            im_l = imread(testImageName);
            
            % read ground truth image
            fn_full_gnd = fullfile(sprintf('Data/Testing/%s_gnd.bmp',image_list{i}(1:end-4)));
            im = imread(fn_full_gnd);
            
            %save image
            srfilename2 = sprintf('%s_%s_res_sr-test_wproj.png',image_list{i}(1:end-4),outstring);
            fn_full_sr2 = fullfile(outdir,srfilename2);
            
%             %save results
%             fmat2 = sprintf('%s/%s_%s_results_SR.mat',outdir,image_list{i}(1:end-4),outstring);
%             save(fmat2,'im_h_d2_wproj','im_b','im_l','im_h_y_1_d2a','im_h_y_2_d2b',...
%                 'im_h_y_d2','im_h_cb',...
%                 'im_h_cr','dictFileName','lambda','patch_size','overlap','up_scale',...
%                 'maxIter','testImageName','stats_d2'); % , 'im_h_y_2',
            disp('');
            %save results
            fstats = sprintf('%s/%s_%s_stats_SR.mat',outdir,image_list{i}(1:end-4),outstring);
            load(fstats,'stats_d2','dictsize','samplesize'); % , 'im_h_y_2',
            disp('');

            allresults_wproj(ss,aa,1) = stats_d2.psnr.sr;
            allresults_wproj(ss,aa,2) = stats_d2.rmse.sr;
            allresults_wproj(ss,aa,3) = stats_d2.ssim.sr;
            allresults_wproj(ss,aa,4) = stats_d2.qssim.sr;

            allresults_bi(ss,aa,1) = stats_d2.psnr.bi;
            allresults_bi(ss,aa,2) = stats_d2.rmse.bi;
            allresults_bi(ss,aa,3) = stats_d2.ssim.bi;
            allresults_bi(ss,aa,4) = stats_d2.qssim.bi;
            
            
            %save image
            srfilename2b = sprintf('%s_res_sr-test_d2_noproj.png',image_list{i}(1:end-4));
            fn_full_sr2b = fullfile(outdir,srfilename2b);
            
%             %save results
%             fmat2 = sprintf('%s/%s_results_SR_d2_noproj.mat',outdir,image_list{i}(1:end-4));
%             save(fmat2,'im_h_d2_noproj','im_b','im_l','im_h_y_1_d2a','im_h_y_2_d2b','im_h_y_d2','im_h_cb',...
%             'im_h_cr','dictFileName','lambda','patch_size','overlap','up_scale',...
%             'maxIter','testImageName','stats_d2_noproj'); % , 'im_h_y_2',
            disp('');
             %save results
            fstats = sprintf('%s/%s_%s_stats_SR_noproj.mat',outdir,image_list{i}(1:end-4),outstring);
            load(fstats,'stats_d2_noproj','dictsize','samplesize','noprojtoc','noprojcputoc','wprojtoc','wprojcputoc'); % , 'im_h_y_2',
            disp('');

            allresults_noproj(ss,aa,1) = stats_d2_noproj.psnr.sr;
            allresults_noproj(ss,aa,2) = stats_d2_noproj.rmse.sr;
            allresults_noproj(ss,aa,3) = stats_d2_noproj.ssim.sr;
            allresults_noproj(ss,aa,4) = stats_d2_noproj.qssim.sr;


% 
% 
%             %% Compare results from {ord2, sm2}, {wproj, noproj}
%     
%             fprintf('%s upscale%d\n\n',outstring,up_scale);
% 
%             fprintf('ScSR(no proj) elap time:           %f s\n', noprojtoc); %bb_psnr);
%             fprintf('ScSR(w proj) elap time:           %f s\n', wprojtoc); %bb_psnr);
%             fprintf('ScSR(no proj) cpu time:           %f s\n', noprojcputoc); %bb_psnr);
%             fprintf('ScSR(w proj) cpu time:           %f s\n', wprojcputoc); %bb_psnr);
% 
%             fprintf('PSNR for Bicubic:          %f dB\n', stats_d2.psnr.bi); %bb_psnr);
%             fprintf('PSNR for ScSR, w proj:     %f dB\n', stats_d2.psnr.sr); %
%             fprintf('PSNR for ScSR, no proj:    %f dB\n', stats_d2_noproj.psnr.sr); %
%             
%             fprintf('RMSE for Bicubic:          %f \n', stats_d2.rmse.bi);
%             fprintf('RMSE for ScSR, w proj:     %f \n', stats_d2.rmse.sr);
%             fprintf('RMSE for ScSR, no proj:    %f \n', stats_d2_noproj.rmse.sr);
%             
%             fprintf('SSIM for Bicubic:          %f \n', stats_d2.ssim.bi);
%             fprintf('SSIM for ScSR, w proj:     %f \n', stats_d2.ssim.sr);
%             fprintf('SSIM for ScSR, no proj:    %f \n', stats_d2_noproj.ssim.sr);
%             
%             fprintf('QSSIM for Bicubic:         %f \n', stats_d2.qssim.bi);
%             fprintf('QSSIM for ScSR, w proj:    %f \n', stats_d2.qssim.sr);
%             fprintf('QSSIM for ScSR, no proj:   %f \n', stats_d2_noproj.qssim.sr);
% 
%             % display stats
%             ftext = sprintf('%s/%s_%s_stats_SR_compare.txt',outdir,image_list{i}(1:end-4),outstring);
%             fid = fopen(ftext,'w');
% 
%             fprintf(fid,'%s upscale%d\n\n',outstring,up_scale);
% 
%             fprintf(fid,'PSNR for Bicubic:          %f dB\n', stats_d2.psnr.bi); %bb_psnr);
%             fprintf(fid,'PSNR for ScSR, w proj:     %f dB\n', stats_d2.psnr.sr); %
%             fprintf(fid,'PSNR for ScSR, no proj:    %f dB\n', stats_d2_noproj.psnr.sr); %
%             
%             fprintf(fid,'RMSE for Bicubic:          %f \n', stats_d2.rmse.bi);
%             fprintf(fid,'RMSE for ScSR, w proj:     %f \n', stats_d2.rmse.sr);
%             fprintf(fid,'RMSE for ScSR, no proj:    %f \n', stats_d2_noproj.rmse.sr);
%             
%             fprintf(fid,'SSIM for Bicubic:          %f \n', stats_d2.ssim.bi);
%             fprintf(fid,'SSIM for ScSR, w proj:     %f \n', stats_d2.ssim.sr);
%             fprintf(fid,'SSIM for ScSR, no proj:    %f \n', stats_d2_noproj.ssim.sr);
%             
%             fprintf(fid,'QSSIM for Bicubic:         %f \n', stats_d2.qssim.bi);
%             fprintf(fid,'QSSIM for ScSR, w proj:    %f \n', stats_d2.qssim.sr);
%             fprintf(fid,'QSSIM for ScSR, no proj:   %f \n', stats_d2_noproj.qssim.sr);
% 
%             fclose(fid);

            

        end



    end
end



array2table(allresults_wproj(:,:,1),'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'})

% 
% dictDir = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/Dictionary_rj_mod_no3';
% dictName = 'D_512_lam-0.15_patchsz-5_zoom-2_Yang.mat';
% dictFileName2 = fullfile(dictDir,dictName);
% load(dictFileName2,'Dh','Dl');
% 
% %[ set parameters
% upscaleFactor = 3;
% patch_size = 5;
% lambda = 0.1;                   % sparsity regularization
% overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
% up_scale = 3;                   % scaling factor, depending on the trained dictionary
% maxIter = 10;                   % if 0, do not use backprojection
% 
% %[ load image
% image_list = {'Lena.png','Child.png','Face.png'};
% % i = 1;
% for i=3 %2:length(image_list)
% 
%     if i==3
%         up_scale = 3; 
%     else
%         up_scale = 2;
%     end
%     
%     % read test image
%     fn_full = fullfile(sprintf('Data/Testing/%s_res.png',image_list{i}(1:end-4)));
%     testImageName = sprintf('Data/Testing/%s',image_list{i});
%     im_l = imread(testImageName);
%     
%     % read ground truth image
%     fn_full_gnd = fullfile(sprintf('Data/Testing/%s_gnd.bmp',image_list{i}(1:end-4)));
%     im = imread(fn_full_gnd);
%     
%     % change color space, work on illuminance only
%     im_l_ycbcr = rgb2ycbcr(im_l);
%     im_l_y = im_l_ycbcr(:, :, 1);
%     im_l_cb = im_l_ycbcr(:, :, 2);
%     im_l_cr = im_l_ycbcr(:, :, 3);
%     
%     
%     %% ScSR - no 3rd order features
%     
%     %[[[ Regular 2nd order features - ScSR
%     % image super-resolution based on sparse representation
%     [im_h_y_1_d2a] = ScSR(im_l_y_noisy, up_scale, Dh2, Dl2, lambda, overlap);
%     if i==3
%         im_h_y_2_d2b=im_h_y_1_d2a;
%     else
%         [im_h_y_2_d2b] = ScSR(im_h_y_1_d2a, up_scale, Dh2, Dl2, lambda, overlap);
%     end
%     [im_h_y_d2] = backprojection(im_h_y_2_d2b, im_l_y_noisy, maxIter);
%     
%     %%%%%%%
%     % Get bicubic + nearest interpolated images
%     [nrow, ncol] = size(im_h_y_d2);
%     % bicubic interpolation for reference
%     im_b = imresize(im_l, [nrow, ncol], 'bicubic');
%     im_b_noisy = imresize(im_l_noisy, [nrow, ncol], 'bicubic');
%     % bicubic interpolation for reference
%     im_l_near = imresize(im_l, [nrow, ncol], 'nearest');
%     im_l_near_noisy = imresize(im_l_noisy, [nrow, ncol], 'nearest');
%     %%%%%%%
%     % upscale the chrominance simply by "bicubic"
%     im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
%     im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
%     
%     %[[[ Regular 2nd order features - with backproj step
%     im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
%     im_h_ycbcr_d2(:, :, 1) = im_h_y_d2;
%     im_h_ycbcr_d2(:, :, 2) = im_h_cb;
%     im_h_ycbcr_d2(:, :, 3) = im_h_cr;
%     im_h_d2_wproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
%     
%     %save image
%     srfilename2 = sprintf('%s_res_sr-test_d2_wproj.png',image_list{i}(1:end-4));
%     fn_full_sr2 = fullfile(outdir,srfilename2);
%     if exist(fn_full_sr2,'file')
%         fprintf(' sr file exists!\n%s\n',fn_full_sr2);
%     end
%     fid = fopen(fn_full_sr2,'w+');
%     fclose(fid);
%     imwrite(im_h_d2_wproj,fn_full_sr2);
%     %compute stats
%     [ stats_d2 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2_wproj );
%     %save results
%     fmat2 = sprintf('%s/%s_results_SR_d2.mat',outdir,image_list{i}(1:end-4));
%     save(fmat2,'im_h_d2_wproj','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b',...
%         'im_h_y_d2','im_h_cb',...
%         'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
%         'maxIter','testImageName','stats_d2'); % , 'im_h_y_2',
%     disp('');
%     
%     
%     %[[[ Regular 2nd order features - skip backproj step
%     im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
%     im_h_ycbcr_d2(:, :, 1) = im_h_y_2_d2b;
%     im_h_ycbcr_d2(:, :, 2) = im_h_cb;
%     im_h_ycbcr_d2(:, :, 3) = im_h_cr;
%     im_h_d2_noproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
%     %compute stats
%     [ stats_d2_noproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d2_noproj );
%     %save image
%     srfilename2b = sprintf('%s_res_sr-test_d2_noproj.png',image_list{i}(1:end-4));
%     fn_full_sr2b = fullfile(outdir,srfilename2b);
%     if exist(fn_full_sr2b,'file')
%     fprintf(' sr file exists!\n%s\n',fn_full_sr2b);
%     end
%     fid = fopen(fn_full_sr2b,'w+');
%     fclose(fid);
%     imwrite(im_h_d2_noproj,fn_full_sr2b);
%     %save results
%     fmat2 = sprintf('%s/%s_results_SR_d2_noproj.mat',outdir,image_list{i}(1:end-4));
%     save(fmat2,'im_h_d2_noproj','im_b_noisy','im_l_noisy','im_h_y_1_d2a','im_h_y_2_d2b','im_h_y_d2','im_h_cb',...
%     'im_h_cr','dictFileName2','lambda','patch_size','overlap','up_scale',...
%     'maxIter','testImageName','stats_d2_noproj'); % , 'im_h_y_2',
%     disp('');
%     
%     
%     
%     
%     %% ScSR - w 3rd order features
%     
%     % image super-resolution based on sparse representation
%     [im_h_y_1_d3a] = ScSR(im_l_y_noisy, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
%     if i==3
%         im_h_y_2_d3b=im_h_y_1_d3a;
%     else
%         [im_h_y_2_d3b] = ScSR(im_h_y_1_d3a, up_scale, Dh3, Dl3, lambda, overlap, '2sm');
%     end
%     [im_h_y_d3] = backprojection(im_h_y_2_d3b, im_l_y_noisy, maxIter);
%     
%     %[[[ Smooth 2nd order features - with backproj step
%     im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
%     im_h_ycbcr_d3(:, :, 1) = im_h_y_d3;
%     im_h_ycbcr_d3(:, :, 2) = im_h_cb;
%     im_h_ycbcr_d3(:, :, 3) = im_h_cr;
%     im_h_d3_wproj = ycbcr2rgb(uint8(im_h_ycbcr_d3));
%     
%     %save image
%     srfilename3 = sprintf('%s_res_sr-test_d3_wproj.png',image_list{i}(1:end-4));
%     fn_full_sr3 = fullfile(outdir,srfilename3);
%     if exist(fn_full_sr3,'file')
%     fprintf(' sr file exists!\n%s\n',fn_full_sr3);
%     end
%     fid = fopen(fn_full_sr3,'w+');
%     fclose(fid);
%     imwrite(im_h_d3_wproj,fn_full_sr3);
%     %compute stats
%     [ stats_d3 ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3_wproj );
%     %save results
%     fmat3 = sprintf('%s/%s_results_SR_d3_wproj.mat',outdir,image_list{i}(1:end-4));
%     save(fmat3,'im_h_d3_wproj','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
%     'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
%     'maxIter','testImageName','stats_d3'); % , 'im_h_y_2',
%     disp('');
%     
%     
%     %[[[ Smooth 2nd order features - no backproj step
%     im_h_ycbcr_d3 = zeros([nrow, ncol, 3]);
%     im_h_ycbcr_d3(:, :, 1) = im_h_y_2_d3b;
%     im_h_ycbcr_d3(:, :, 2) = im_h_cb;
%     im_h_ycbcr_d3(:, :, 3) = im_h_cr;
%     im_h_d3_noproj = ycbcr2rgb(uint8(im_h_ycbcr_d3));
%     
%     %save image
%     srfilename3b = sprintf('%s_res_sr-test_d3_noproj.png',image_list{i}(1:end-4));
%     fn_full_sr3b = fullfile(outdir,srfilename3b);
%     if exist(fn_full_sr3b,'file')
%     fprintf(' sr file exists!\n%s\n',fn_full_sr3b);
%     end
%     fid = fopen(fn_full_sr3b,'w+');
%     fclose(fid);
%     imwrite(im_h_d3_noproj,fn_full_sr3b);
%     %compute stats
%     [ stats_d3_noproj ] = compute_image_quality_metrics( im, im_b_noisy, im_h_d3_noproj );
%     %save results
%     fmat3 = sprintf('%s/%s_results_SR_d3_noproj.mat',outdir,image_list{i}(1:end-4));
%     save(fmat3,'im_h_d3_noproj','im_b','im_l','im_b_noisy','im_l_noisy','im_h_y_1_d3a','im_h_y_2_d3b','im_h_y_d3','im_h_cb',...
%     'im_h_cr','dictFileName3','lambda','patch_size','overlap','up_scale',...
%     'maxIter','testImageName','stats_d3_noproj'); % , 'im_h_y_2',
%     disp('');
    
    
    
    
    
%     %% Compare results from {ord2, sm2}, {wproj, noproj}
%     
%     % display stats
%     fprintf('\n');
%     fprintf('PSNR for Bicubic:          %f dB\n', stats_d2.psnr.bi); %bb_psnr);
%     fprintf('PSNR for ScSR, w proj:     %f dB\n', stats_d2.psnr.sr); %
%     fprintf('PSNR for SmScSR, w proj:   %f dB\n', stats_d3.psnr.sr); %
%     fprintf('PSNR for ScSR, no proj:    %f dB\n', stats_d2_noproj.psnr.sr); %
%     fprintf('PSNR for SmScSR, no proj:  %f dB\n\n', stats_d3_noproj.psnr.sr); %
%     
%     fprintf('RMSE for Bicubic:          %f \n', stats_d2.rmse.bi);
%     fprintf('RMSE for ScSR, w proj:     %f \n', stats_d2.rmse.sr);
%     fprintf('RMSE for SmScSR, w proj:   %f \n', stats_d3.rmse.sr);
%     fprintf('RMSE for ScSR, no proj:    %f \n', stats_d2_noproj.rmse.sr);
%     fprintf('RMSE for SmScSR, no proj:  %f \n\n', stats_d3_noproj.rmse.sr);
%     
%     fprintf('SSIM for Bicubic:          %f \n', stats_d2.ssim.bi);
%     fprintf('SSIM for ScSR, w proj:     %f \n', stats_d2.ssim.sr);
%     fprintf('SSIM for SmScSR, w proj:   %f \n', stats_d3.ssim.sr);
%     fprintf('SSIM for ScSR, no proj:    %f \n', stats_d2_noproj.ssim.sr);
%     fprintf('SSIM for SmScSR, no proj:  %f \n\n', stats_d3_noproj.ssim.sr);
%     
%     fprintf('QSSIM for Bicubic:         %f \n', stats_d2.qssim.bi);
%     fprintf('QSSIM for ScSR, w proj:    %f \n', stats_d2.qssim.sr);
%     fprintf('QSSIM for SmScSR, w proj:  %f \n', stats_d3.qssim.sr);
%     fprintf('QSSIM for ScSR, no proj:   %f \n', stats_d2_noproj.qssim.sr);
%     fprintf('QSSIM for SmScSR, no proj: %f \n\n', stats_d3_noproj.qssim.sr);
    