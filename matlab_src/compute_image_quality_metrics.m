function [ stats ] = compute_image_quality_metrics( im, im_b, im_h )
    % im = ground truth HR
    % im_b = blinear HR
    % im_h = superres HR
    
    stats.arr = zeros(4,2);
    stats = [];

    % compute PSNR for the illuminance channel
    bb_rmse = compute_rmse(im, im_b);
    sp_rmse = compute_rmse(im, im_h);
    stats.rmse.bi = bb_rmse;
    stats.rmse.sr = sp_rmse;

    [qssim_sp,~] = qssim(im, im_h);
    [qssim_in,~] = qssim(im, im_b);
    stats.qssim.bi = qssim_in;
    stats.qssim.sr = qssim_sp;
    
    im_gray = rgb2gray(im);
    im_h_gray = rgb2gray(im_h);
    im_b_gray = rgb2gray(im_b);
    [ssim_sp,~] = ssim_index(im_gray,im_h_gray);
    [ssim_in,~] = ssim_index(im_gray,im_b_gray);
    stats.ssim.bi = ssim_in;
    stats.ssim.sr = ssim_sp;
    
    bb_psnr = 20*log10(255/bb_rmse);
    sp_psnr = 20*log10(255/sp_rmse);

    stats.psnr.bi = bb_psnr;
    stats.psnr.sr = sp_psnr;

    
    fprintf('\n');
    fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
    fprintf('PSNR for Sparse Representation Recovery: %f dB\n\n', sp_psnr);

    fprintf('RMSE for Bicubic Interpolation: %f dB\n', bb_rmse);
    fprintf('RMSE for Sparse Representation Recovery: %f dB\n\n', sp_rmse);

    fprintf('SSIM for Bicubic Interpolation: %f dB\n', ssim_in);
    fprintf('SSIM for Sparse Representation Recovery: %f dB\n\n', ssim_sp);

    fprintf('QSSIM for Bicubic Interpolation: %f dB\n', qssim_in);
    fprintf('QSSIM for Sparse Representation Recovery: %f dB\n\n', qssim_sp);
    
    
%     % show the images
%     figure, 
%     subplot(121);
%     imshow(im_h);
%     title('Sparse Recovery');
%     subplot(122);
%     imshow(im_b);
%     title('Bicubic Interpolation');

%     fmat = sprintf('Results-Testing/%s_res_sr-metrics.mat',image_list{i}(1:end-4));
%     save(fmat,'bb_rmse','sp_rmse','qssim_sp','qssim_in','ssim_in','ssim_sp','bb_psnr','sp_psnr')
end

