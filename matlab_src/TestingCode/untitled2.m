
addpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/sparseCodingSuperResolution-master');
addpath('/Users/robertjones/Desktop/W24/556/project/super-res-sparse-coding/qtfm');


outroot = '/Users/robertjones/Desktop/W24/556/project/progress-figs/tmp';
mkdir(outroot);

fnum = '89';
% /Users/robertjones/Downloads/Archive 2/512_US3_L0.1_PS5bilateral_42012
switch fnum
    case '89'
        roix = 71:170;
        roiy = 106:205;
end
% /Users/robertjones/Downloads/512_US3_L0.1_PS5_brain
% fdir = ['/Users/robertjones/Downloads/2048_US3_L0.1_PS3_' fnum];
fdir = ['/Users/robertjones/Downloads/tmp'];
outdir = [outroot filesep fnum];
if ~exist(outdir,'dir'), mkdir(outdir); end

%% load images
lrorig = imread([fdir filesep 'LR.png']);
br = imread([fdir filesep '3bilateral.png']);
hr = imread([fdir filesep '3HR.png']);
sr = imread([fdir filesep '2SR.png']);
cr = imread([fdir filesep '4cnn.png']);
rr = imread([fdir filesep '5recon.png']);
cr = cr(12:end-12,12:end-12,:);
cr = imresize(cr,size(hr,1:2));

%% compute metrics

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(hr, br);
sp_rmse = compute_rmse(hr, sr);
cnn_rmse = compute_rmse(hr, cr);
rec_rmse = compute_rmse(hr, rr);

[qssim_sp,~] = qssim(hr, sr);
[qssim_in,~] = qssim(hr, br);

im_gray = rgb2gray(hr);
im_h_gray = rgb2gray(sr);
im_b_gray = rgb2gray(br);
im_c_gray = rgb2gray(cr);
im_r_gray = rgb2gray(rr);

[ssim_sp,~] = ssim_index(im_gray,im_h_gray);
[ssim_in,~] = ssim_index(im_gray,im_b_gray);
[ssim_cnn,~] = ssim_index(im_gray,im_c_gray);
[ssim_rec,~] = ssim_index(im_gray,im_r_gray);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);
cnn_psnr = 20*log10(255/cnn_rmse);
rec_psnr = 20*log10(255/rec_rmse);

fprintf('\n');
fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);
fprintf('PSNR for CNN Recovery: %f dB\n', cnn_psnr);
fprintf('PSNR for rec Recovery: %f dB\n\n', rec_psnr);

fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
fprintf('RMSE for Sparse Representation Recovery: %f dB\n', sp_rmse);
fprintf('RMSE for cnn Recovery: %f dB\n', cnn_rmse);
fprintf('RMSE for rec Recovery: %f dB\n\n', rec_rmse);

fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_in);
fprintf('SSIM for Sparse Representation Recovery: %f dB\n', ssim_sp);
fprintf('SSIM for cnn Recovery: %f dB\n', ssim_cnn);
fprintf('SSIM for rec Recovery: %f dB\n\n', ssim_rec);

fprintf('QSSIM for Bicubic Interpolation: %f dB\n', qssim_in);
fprintf('QSSIM for Sparse Representation Recovery: %f dB\n\n', qssim_sp);
    
fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
fprintf('RMSE for Sparse Representation Recovery: %f dB\n', sp_rmse);
fprintf('RMSE for cnn Recovery: %f dB\n', 7.57637501593893);
fprintf('RMSE for rec Recovery: %f dB\n\n', rec_rmse);

%     7.57637501593893
    % show the images

%% make figs
% add red box to LR image
hrdims = size(hr);
lr = imresize(lrorig,hrdims(1:2),"nearest");
lrmod = lr;
 lrmod(roix(1):roix(end),roiy(1),1)=255;
lrmod(roix(1):roix(end),roiy(1),2)=0;
lrmod(roix(1):roix(end),roiy(1),3)=0;
 lrmod(roix(1):roix(end),roiy(end),1)=255;
lrmod(roix(1):roix(end),roiy(end),2)=0;
lrmod(roix(1):roix(end),roiy(end),3)=0;
 lrmod(roix(1),roiy(1):roiy(end),1)=255;
lrmod(roix(1),roiy(1):roiy(end),2)=0;
lrmod(roix(1),roiy(1):roiy(end),3)=0;
 lrmod(roix(end),roiy(1):roiy(end),1)=255;
lrmod(roix(end),roiy(1):roiy(end),2)=0;
lrmod(roix(end),roiy(1):roiy(end),3)=0;

% fig = figure('color','w','position',[88 312 1425 540]);
% fontsize(fig, 24, "points")
% tiledlayout(1,5);
% nexttile;
% imshow(lrmod); title('LR');
% fontsize(gca, 40, "points")
% nexttile;
% imshow(hr); title('HR');
% fontsize(gca, 40, "points")
% nexttile;
% imshow(sr); title('SR');
% fontsize(gca, 40, "points")
% nexttile;
% imshow(br); title('Bilateral');
% fontsize(gca, 40, "points")
% nexttile;
% imshow(cr); title('Learnt Feature Map');
% fontsize(gca, 40, "points")

fig = figure('color','w','position',[7 406 1506 446]);
fontsize(fig, 24, "points")
tiledlayout(1,5);
nexttile;
imshow(hr); title('HR');
fontsize(gca, 40, "points")
nexttile;
imshow(br); title('Bilateral');
fontsize(gca, 40, "points")
nexttile;
imshow(sr); title('SR');
fontsize(gca, 40, "points")
nexttile;
imshow(rr); title({'Reconstruction','Constraint'});
fontsize(gca, 30, "points")
nexttile;
imshow(cr); title('Learnt');
fontsize(gca, 40, "points")


print(gcf,[outdir filesep 'sr-montage.png'],'-dpng','-r300');



cropped.lr = lr(roix,roiy,:);
cropped.hr = hr(roix,roiy,:);
cropped.sr = sr(roix,roiy,:);
cropped.br = br(roix,roiy,:);
cropped.rr = rr(roix,roiy,:);
cropped.cr = cr(roix,roiy,:);

fcroplr = [outdir filesep 'crop-lr.png'];
imwrite(cropped.lr,fcroplr)
fcrophr = [outdir filesep 'crop-hr.png'];
imwrite(cropped.hr,fcrophr)
fcropsr = [outdir filesep 'crop-sr.png'];
imwrite(cropped.sr,fcropsr)
fcropbr = [outdir filesep 'crop-br.png'];
imwrite(cropped.br,fcropbr)
fcroprr = [outdir filesep 'crop-rr.png'];
imwrite(cropped.br,fcroprr)
fcropcr = [outdir filesep 'crop-cr.png'];
imwrite(cropped.br,fcropcr)


fig = figure('color','w','position',[10 316 1475 536]);
fontsize(fig, 24, "points")
tiledlayout(1,5);
nexttile;
imshow(cropped.hr); 
% title('HR');
% fontsize(gca, 24, "points")
nexttile;
imshow(cropped.sr);
% title('SR');
% fontsize(gca, 24, "points")
nexttile;
imshow(cropped.cr); 
% title('Bicubic');
% fontsize(gca, 24, "points")

print(gcf,[outdir filesep 'sr-montage-zoom.png'],'-dpng','-r300');


