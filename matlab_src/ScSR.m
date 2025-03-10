function [hIm] = ScSR(lIm, up_scale, Dh, Dl, lambda, overlap, featType)
% [hIm] = ScSR(lIm, up_scale, Dh, Dl, lambda, overlap, featType)
%   Run ScSR single-image super-resolution restoration/reconstruction.
% 
% INPUTS:
%   lIm       = input LR image to recon
%   up_scale  = magnification factor 
%   Dh        = HR dict     
%   Dl        = LR dict   
%   lambda    = sparsity regularization param 
%   overlap   = overlap (pixels) b/w adjacent patches
%   featType  = LR feature type used in dict; (optional;)
%                2==normal Yang et al 2010 (default)
%                'sm2','2sm' == smoothed second order 
%                3 == third order features included
% OUTPUTS:
%   hIm       = Super-resolution restored image
%
%


if nargin<7
    featType = 2;
end

% normalize the dictionary
norm_Dl = sqrt(sum(Dl.^2, 1)); 
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);

patch_size = sqrt(size(Dh, 1));

% bicubic interpolation of the low-resolution image
mIm = single(imresize(lIm, up_scale, 'bicubic'));

hIm = zeros(size(mIm));
cntMat = zeros(size(mIm));

[h, w] = size(mIm);

% extract low-resolution image features
if featType == 2
    lImfea = extr_lIm_fea(mIm);
elseif featType == 3
    lImfea = extr_lIm_fea_mod(mIm);
elseif contains(featType,'2sm') || contains(featType,'sm2')
    lImfea = extr_lIm_fea_sm2(mIm);
end

% patch indexes for sparse recovery (avoid boundary)
gridx = 3:patch_size - overlap : w-patch_size-2;
gridx = [gridx, w-patch_size-2];
gridy = 3:patch_size - overlap : h-patch_size-2;
gridy = [gridy, h-patch_size-2];

A = Dl'*Dl;
cnt = 0;

% loop to recover each low-resolution patch
for ii = 1:length(gridx),
    for jj = 1:length(gridy),
        num = (ii-1)*length(gridy)+jj;
        if mod(num,1000) == 0
          fprintf('%d---%d\n',num, length(gridx)*length(gridy));
        end
        cnt = cnt+1;
        xx = gridx(ii);
        yy = gridy(jj);
        
        mPatch = mIm(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean = mean(mPatch(:));
        mPatch = mPatch(:) - mMean;
        mNorm = sqrt(sum(mPatch.^2));
        
        mPatchFea = lImfea(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
        mPatchFea = mPatchFea(:);
        mfNorm = sqrt(sum(mPatchFea.^2));
        
        if mfNorm > 1,
            y = mPatchFea./mfNorm;
        else
            y = mPatchFea;
        end
        
        b = -Dl'*y;
      
        % sparse recovery
        w = L1QP_FeatureSign_yang(lambda, A, b);
        
        % generate the high resolution patch and scale the contrast
        hPatch = Dh*w;
        hPatch = lin_scale(hPatch, mNorm);
        
        hPatch = reshape(hPatch, [patch_size, patch_size]);
        hPatch = hPatch + mMean;
        
        hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch;
        cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
    end
end

% fill in the empty with bicubic interpolation
idx = (cntMat < 1);
hIm(idx) = mIm(idx);

cntMat(idx) = 1;
hIm = hIm./cntMat;
hIm = uint8(hIm);
