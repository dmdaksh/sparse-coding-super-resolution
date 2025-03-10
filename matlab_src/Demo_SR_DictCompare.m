function [ allresults ] = Demo_SR_DictCompare( dictroot, outdir )
% [ allresults ] = Demo_SR_DictCompare( dictroot, outdir )
%  dictroot = root directory where dictionaries are saved (from
%                   submit_DictionaryTraining_comparison.m)
%         - Here, dictionaries are saved in sub-directories {25k,50k,100k}
%           based on # of randomly sampled patches.
% outdir    = path to save super-resolution results and stats 
% 
% 


cd /Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src

addpath('RegularizedSC/sc2');
addpath('RegularizedSC');
addpath('utils');
addpath('qtfm');



%% ScSR - dictionary comparison
if nargin==0
    outdir = 'New-Results-dict-compare';
    if ~exist(outdir,'dir'), mkdir(outdir); end
    
    dictroot = '/Users/robertjones/Desktop/W24/556/project/sparse-coding-super-resolution/matlab_src/newdicts';
end
samplesizes = [25,50,100];
nsamplesizes = length(samplesizes);

dictsizes = [256,512,1024];
ndictsizes = length(dictsizes);

%[ set parameters
patch_size = 5;
lambda = 0.1;                   % sparsity regularization
overlap = patch_size - 1;                    % the more overlap the better (patch size 5x5)
up_scale = 3;                   % scaling factor, depending on the trained dictionary
maxIter = 10;                   % if 0, do not use backprojection

%[ List of test images to run ScSR recons on:
% The Face image has a LR image and a HR image upscaled by magnification factor 3
image_list = {'Face.png'};

% Matrices to store image quality metrics
nmetrics = 4; %[PSNR, RMSE, SSIM, QSSIM]
allresults_bi = zeros(nsamplesizes,ndictsizes,nmetrics);
allresults_scsr_noproj = zeros(nsamplesizes,ndictsizes,nmetrics);
allresults_scsr_wproj = zeros(nsamplesizes,ndictsizes,nmetrics);

% Store the patsh to dictionaries used, for reference
allDictFiles = cell(nsamplesizes,ndictsizes);

% Loop through dict params and run ScSR recons
for ss=1:nsamplesizes
    samplesize = samplesizes(ss);

    for aa=1:ndictsizes
        dictsize = dictsizes(aa);
        
        dictDir = fullfile(dictroot,sprintf('%dk',samplesize));
        outstring = sprintf('nsamples%d-dictsize%d',samplesize,dictsize);

        % Load dictionaries to use
        dictName = sprintf('D_%d_lam-0.1_patchsz-5_zoom-3_Yang.mat',dictsize);
        dictFileName = fullfile(dictDir,dictName);
        allDictFiles{ss,aa} = dictFileName;
        load(dictFileName,'Dh','Dl');

        % recon test images
        for i=1:length(image_list)
            
            % read LR test image
            testImageName = sprintf('Data/Testing/%s',image_list{i});
            im_l = imread(testImageName);
            
            % read HR ground truth test image
            fn_full_gnd = fullfile(sprintf('Data/Testing/%s_gnd.bmp',image_list{i}(1:end-4)));
            im = imread(fn_full_gnd);
            
            % change color space, work on illuminance only
            im_l_ycbcr = rgb2ycbcr(im_l);
            im_l_y = im_l_ycbcr(:, :, 1);
            im_l_cb = im_l_ycbcr(:, :, 2);
            im_l_cr = im_l_ycbcr(:, :, 3);
            
            
            %% ScSR reconstruction
            
            atic = tic;
            acputic = cputime;

            %[[[ ScSR super-resolution based on sparse representation
            [im_h_y_1_d2a] = ScSR(im_l_y, up_scale, Dh, Dl, lambda, overlap);
            im_h_y_2_d2b = im_h_y_1_d2a; % ScSR before backprojection

            noprojtoc = toc(atic);
            noprojcputoc = cputime - acputic;

            % Backprojection to enforce global reconstruction constraint
            [im_h_y_d2] = backprojection(im_h_y_2_d2b, im_l_y, maxIter);
                % ScSR after backprojection

            wprojtoc = toc(atic);
            wprojcputoc = cputime - acputic;

            %%%%%%%
            % Get bicubic + nearest interpolated images
            [nrow, ncol] = size(im_h_y_d2);
            % bicubic interpolation for reference
            im_b = imresize(im_l, [nrow, ncol], 'bicubic');
            % bicubic interpolation for reference
            im_l_near = imresize(im_l, [nrow, ncol], 'nearest');
            %%%%%%%
            % upscale the chrominance simply by "bicubic"
            im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
            im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

            %% [ Get restored ScSR HR image (using back-projection)
            
            %[[[ form HR image in YCbCr format + convert to RGB
            im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
            im_h_ycbcr_d2(:, :, 1) = im_h_y_d2;
            im_h_ycbcr_d2(:, :, 2) = im_h_cb;
            im_h_ycbcr_d2(:, :, 3) = im_h_cr;
            im_h_wproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
            
            %save image
            srfilename2 = sprintf('%s_%s_res_sr-test_wproj.png',image_list{i}(1:end-4),outstring);
            fn_full_sr2 = fullfile(outdir,srfilename2);
            if exist(fn_full_sr2,'file')
                fprintf(' warning: sr file exists!\n%s\n',fn_full_sr2);
                fprintf(' (pausing for 5seconds... will overwrite if not killed\n');
                pause(5);
            end
            fid = fopen(fn_full_sr2,'w+');
            fclose(fid);
            imwrite(im_h_wproj,fn_full_sr2);
            %compute image quality stats (RMSE, PSNR, SSIM, QSSIM)
            [ stats_wproj ] = compute_image_quality_metrics( im, im_b, im_h_wproj );
            %save results
            fmat2 = sprintf('%s/%s_%s_results_SR.mat',outdir,image_list{i}(1:end-4),outstring);
            save(fmat2,'im_h_wproj','im_b','im_l','dictFileName','lambda',...
                'patch_size','overlap','up_scale','maxIter','testImageName','stats_wproj');
            %save stats results
            fstats = sprintf('%s/%s_%s_stats_SR.mat',outdir,image_list{i}(1:end-4),outstring);
            save(fstats,'stats_wproj','dictsize','samplesize','wprojtoc','wprojcputoc'); % , 'im_h_y_2',

            % Store IQ metrics for ScSR, with back-projection
            allresults_scsr_wproj(ss,aa,1) = stats_wproj.psnr.sr;
            allresults_scsr_wproj(ss,aa,2) = stats_wproj.rmse.sr;
            allresults_scsr_wproj(ss,aa,3) = stats_wproj.ssim.sr;
            allresults_scsr_wproj(ss,aa,4) = stats_wproj.qssim.sr;

            % Store IQ metrics for bicubic interpolation
            allresults_bi(ss,aa,1) = stats_wproj.psnr.bi;
            allresults_bi(ss,aa,2) = stats_wproj.rmse.bi;
            allresults_bi(ss,aa,3) = stats_wproj.ssim.bi;
            allresults_bi(ss,aa,4) = stats_wproj.qssim.bi;            
            
            %% [ Get restored ScSR HR image (without using back-projection)
            
            %[[[ form HR image in YCbCr format + convert to RGB
            im_h_ycbcr_d2 = zeros([nrow, ncol, 3]);
            im_h_ycbcr_d2(:, :, 1) = im_h_y_2_d2b;
            im_h_ycbcr_d2(:, :, 2) = im_h_cb;
            im_h_ycbcr_d2(:, :, 3) = im_h_cr;
            im_h_noproj = ycbcr2rgb(uint8(im_h_ycbcr_d2));
            %compute stats
            [ stats_noproj ] = compute_image_quality_metrics( im, im_b, im_h_noproj );
            %save image
            srfilename2b = sprintf('%s_res_sr-test_noproj.png',image_list{i}(1:end-4));
            fn_full_sr2b = fullfile(outdir,srfilename2b);
            if exist(fn_full_sr2b,'file')
                fprintf(' sr file exists!\n%s\n',fn_full_sr2b);
                fprintf(' (pausing for 5seconds... will overwrite if not killed\n');
                pause(5);
            end
            fid = fopen(fn_full_sr2b,'w+');
            fclose(fid);
            imwrite(im_h_noproj,fn_full_sr2b);
            %save results
            fmat2 = sprintf('%s/%s_results_SR_d2_noproj.mat',outdir,image_list{i}(1:end-4));
            save(fmat2,'im_h_noproj','im_b','im_l','dictFileName','lambda',...
                'patch_size','overlap','up_scale','maxIter','testImageName','stats_noproj'); 
             %save stats results
            fstats = sprintf('%s/%s_%s_stats_SR_noproj.mat',outdir,image_list{i}(1:end-4),outstring);
            save(fstats,'stats_noproj','dictsize','samplesize','noprojtoc','noprojcputoc'); % , 'im_h_y_2',

            % Store IQ metrics for ScSR, no-back-projection
            allresults_scsr_noproj(ss,aa,1) = stats_noproj.psnr.sr;
            allresults_scsr_noproj(ss,aa,2) = stats_noproj.rmse.sr;
            allresults_scsr_noproj(ss,aa,3) = stats_noproj.ssim.sr;
            allresults_scsr_noproj(ss,aa,4) = stats_noproj.qssim.sr;


            %% Compare results from {ord2, sm2}, {wproj, noproj}
    
            fprintf('%s upscale%d\n\n',outstring,up_scale);

            fprintf('ScSR(no proj) elap time:           %f s\n', noprojtoc); %bb_psnr);
            fprintf('ScSR(w proj) elap time:           %f s\n', wprojtoc); %bb_psnr);
            fprintf('ScSR(no proj) cpu time:           %f s\n', noprojcputoc); %bb_psnr);
            fprintf('ScSR(w proj) cpu time:           %f s\n', wprojcputoc); %bb_psnr);

            fprintf('PSNR for Bicubic:          %f dB\n', stats_wproj.psnr.bi); %bb_psnr);
            fprintf('PSNR for ScSR, w proj:     %f dB\n', stats_wproj.psnr.sr); %
            fprintf('PSNR for ScSR, no proj:    %f dB\n', stats_noproj.psnr.sr); %
            
            fprintf('RMSE for Bicubic:          %f \n', stats_wproj.rmse.bi);
            fprintf('RMSE for ScSR, w proj:     %f \n', stats_wproj.rmse.sr);
            fprintf('RMSE for ScSR, no proj:    %f \n', stats_noproj.rmse.sr);
            
            fprintf('SSIM for Bicubic:          %f \n', stats_wproj.ssim.bi);
            fprintf('SSIM for ScSR, w proj:     %f \n', stats_wproj.ssim.sr);
            fprintf('SSIM for ScSR, no proj:    %f \n', stats_noproj.ssim.sr);
            
            fprintf('QSSIM for Bicubic:         %f \n', stats_wproj.qssim.bi);
            fprintf('QSSIM for ScSR, w proj:    %f \n', stats_wproj.qssim.sr);
            fprintf('QSSIM for ScSR, no proj:   %f \n', stats_noproj.qssim.sr);

            % save stats to text file
            ftext = sprintf('%s/%s_%s_stats_SR_compare.txt',outdir,image_list{i}(1:end-4),outstring);
            fid = fopen(ftext,'w');

            fprintf(fid,'%s upscale%d\n\n',outstring,up_scale);

            fprintf(fid,'PSNR for Bicubic:          %f dB\n', stats_wproj.psnr.bi); %bb_psnr);
            fprintf(fid,'PSNR for ScSR, w proj:     %f dB\n', stats_wproj.psnr.sr); %
            fprintf(fid,'PSNR for ScSR, no proj:    %f dB\n', stats_noproj.psnr.sr); %
            
            fprintf(fid,'RMSE for Bicubic:          %f \n', stats_wproj.rmse.bi);
            fprintf(fid,'RMSE for ScSR, w proj:     %f \n', stats_wproj.rmse.sr);
            fprintf(fid,'RMSE for ScSR, no proj:    %f \n', stats_noproj.rmse.sr);
            
            fprintf(fid,'SSIM for Bicubic:          %f \n', stats_wproj.ssim.bi);
            fprintf(fid,'SSIM for ScSR, w proj:     %f \n', stats_wproj.ssim.sr);
            fprintf(fid,'SSIM for ScSR, no proj:    %f \n', stats_noproj.ssim.sr);
            
            fprintf(fid,'QSSIM for Bicubic:         %f \n', stats_wproj.qssim.bi);
            fprintf(fid,'QSSIM for ScSR, w proj:    %f \n', stats_wproj.qssim.sr);
            fprintf(fid,'QSSIM for ScSR, no proj:   %f \n', stats_noproj.qssim.sr);

            fclose(fid);

        end

    end
end

allresults.readme = 'Rows = # of patches (25k, 50k, 100k); Cols = # of Dict Atoms (256, 512, 1024)';
allresults.readme = strcat(allresults.readme,' allresults: 3rd dim = [RMSE, PSNR, SSIM, QSSIM]');
allresults.scsr_wproj = allresults_scsr_wproj;
allresults.scsr_noproj = allresults_scsr_noproj;
allresults.bi = allresults_bi;


scsr_wproj_rmse_table = array2table(allresults_scsr_wproj(:,:,1),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_wproj_rmse_table.Properties.Description = 'RMSE (ScSR, with back-projection)';

scsr_wproj_psnr_table = array2table(allresults_scsr_wproj(:,:,2),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_wproj_psnr_table.Properties.Description = 'PSNR (ScSR, with back-projection)';

scsr_wproj_ssim_table = array2table(allresults_scsr_wproj(:,:,3),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_wproj_ssim_table.Properties.Description = 'SSIM (ScSR, with back-projection)';

scsr_wproj_qssim_table = array2table(allresults_scsr_wproj(:,:,4),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_wproj_qssim_table.Properties.Description = 'QSSIM (ScSR, with back-projection)';

fprintf('\n\n [rows = # of randomly sampled patches; cols = # of dict atoms]\n');

fprintf('\n --- %s ---\n',scsr_wproj_rmse_table.Properties.Description);
head(scsr_wproj_rmse_table);

fprintf('\n --- %s ---\n',scsr_wproj_psnr_table.Properties.Description);
head(scsr_wproj_psnr_table);

fprintf('\n --- %s ---\n',scsr_wproj_ssim_table.Properties.Description);
head(scsr_wproj_ssim_table);

fprintf('\n --- %s ---\n',scsr_wproj_qssim_table.Properties.Description);
head(scsr_wproj_qssim_table);



%% Stats, ScSR no backprojection
scsr_noproj_rmse_table = array2table(allresults_scsr_noproj(:,:,1),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_noproj_rmse_table.Properties.Description = 'RMSE (ScSR, no back-projection)';

scsr_noproj_psnr_table = array2table(allresults_scsr_noproj(:,:,2),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_noproj_psnr_table.Properties.Description = 'PSNR (ScSR, no back-projection)';

scsr_noproj_ssim_table = array2table(allresults_scsr_noproj(:,:,3),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_noproj_ssim_table.Properties.Description = 'SSIM (ScSR, no back-projection)';

scsr_noproj_qssim_table = array2table(allresults_scsr_noproj(:,:,4),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
scsr_noproj_qssim_table.Properties.Description = 'QSSIM (ScSR, no back-projection)';

fprintf('\n\n [rows = # of randomly sampled patches; cols = # of dict atoms]\n');

fprintf('\n --- %s ---\n',scsr_noproj_rmse_table.Properties.Description);
head(scsr_noproj_rmse_table);

fprintf('\n --- %s ---\n',scsr_noproj_psnr_table.Properties.Description);
head(scsr_noproj_psnr_table);

fprintf('\n --- %s ---\n',scsr_noproj_ssim_table.Properties.Description);
head(scsr_noproj_ssim_table);

fprintf('\n --- %s ---\n',scsr_noproj_qssim_table.Properties.Description);
head(scsr_noproj_qssim_table);



%% Stats, bicubic
bi_rmse_table = array2table(allresults_bi(:,:,1),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
bi_rmse_table.Properties.Description = 'RMSE (bicubic)';

bi_psnr_table = array2table(allresults_bi(:,:,2),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
bi_psnr_table.Properties.Description = 'PSNR (bicubic)';

bi_ssim_table = array2table(allresults_bi(:,:,3),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
bi_ssim_table.Properties.Description = 'SSIM (bicubic)';

bi_qssim_table = array2table(allresults_bi(:,:,4),...
    'VariableNames',{'256','512','1024'},'RowNames',{'25k','50k','100k'}, ...
    'DimensionNames',{'NbPatches','NbDictAtoms'});
bi_qssim_table.Properties.Description = 'QSSIM (bicubic)';

fprintf('\n\n [rows = # of randomly sampled patches; cols = # of dict atoms]\n');

fprintf('\n --- %s ---\n',bi_rmse_table.Properties.Description);
head(bi_rmse_table);

fprintf('\n --- %s ---\n',bi_psnr_table.Properties.Description);
head(bi_psnr_table);

fprintf('\n --- %s ---\n',bi_ssim_table.Properties.Description);
head(bi_ssim_table);

fprintf('\n --- %s ---\n',bi_qssim_table.Properties.Description);
head(bi_qssim_table);



