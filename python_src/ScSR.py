import numpy as np


import torch
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate


# from os import listdir
# from sklearn.preprocessing import normalize
# from skimage.io import imread
# from skimage.color import rgb2ycbcr
from skimage.transform import resize

# import pickle
from featuresign import fss_yang
from scipy.signal import convolve2d
from tqdm import tqdm

#
def extract_lr_feat(img_lr):
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = [
        [-1, 0, 1],
    ] * 3
    vf1 = np.transpose(hf1)

    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, "same")
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, "same")

    # Second order gradient filters
    hf2 = [
        [1, 0, -2, 0, 1],
    ] * 3
    vf2 = np.transpose(hf2)

    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, "same")
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, "same")

    return img_lr_feat


def extract_cnn_features(y_channel):
    # Load a pretrained ResNet model with the best available pretrained weights
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # Modify the first convolutional layer to use only every 8th filter
    original_first_conv = model.conv1
    new_filters = original_first_conv.weight[::8, :, :, :]
    new_out_channels = new_filters.size(0)

    # Create a new convolutional layer with the selected filters
    model.conv1 = torch.nn.Conv2d(
        in_channels=original_first_conv.in_channels,
        out_channels=new_out_channels,
        kernel_size=original_first_conv.kernel_size,
        stride=original_first_conv.stride,
        padding=original_first_conv.padding,
        bias=(original_first_conv.bias is not None),
    )
    model.conv1.weight = torch.nn.Parameter(new_filters)
    if original_first_conv.bias is not None:
        model.conv1.bias = torch.nn.Parameter(original_first_conv.bias[::8])

    # Adjust the first batch normalization layer
    original_bn = model.bn1
    model.bn1 = torch.nn.BatchNorm2d(new_out_channels)
    model.bn1.weight = torch.nn.Parameter(original_bn.weight[::8])
    model.bn1.bias = torch.nn.Parameter(original_bn.bias[::8])
    model.bn1.running_mean = original_bn.running_mean[::8]
    model.bn1.running_var = original_bn.running_var[::8]

    # Using only the first layers including the modified conv1 and bn1
    first_layer = torch.nn.Sequential(
        *(list(model.children())[:4])
    )  # Include until the first pooling to preserve spatial dimensions
    model = torch.nn.Sequential(first_layer)
    model.eval()  # Set the model to evaluation mode

    # Define the preprocessing transformation for Y channel
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),  # Repeat Y channel to create three channels
            transforms.Normalize(
                mean=weights.transforms().mean, std=weights.transforms().std
            ),
        ]
    )

    # Process the Y channel
    img_tensor = transform(y_channel).unsqueeze(0)  # Add a batch dimension
    img_tensor = img_tensor.float()  # Ensure the tensor is of type torch.float32

    # Get the original image size (height, width)
    original_size = y_channel.size

    # Pass the Y channel through the model
    with torch.no_grad():
        features = model(img_tensor)

    # Resize feature maps to the size of the original Y channel
    features_resized = interpolate(
        features, size=original_size, mode="bilinear", align_corners=False
    )

    # Convert to numpy array for further processing if necessary
    features_np = features_resized.squeeze(0).permute(1, 2, 0).numpy()

    return features_np


def lin_scale(xh, us_norm):
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))

    if hr_norm > 0:
        s = us_norm * 1.2 / hr_norm
        xh = np.multiply(xh, s)
    return xh


def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap, patch_size):
    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_hr)
    # img_lr_y_feat = extract_cnn_features(img_hr)

    ### discard this code ###
    # gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    # gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)
    #########################

    # optimize gridx and gridy  calculation
    gridx = np.arange(0, img_us_width - patch_size - 1, patch_size - overlap)
    gridx = np.append(gridx, img_us_width - patch_size - 1)

    gridy = np.arange(0, img_us_height - patch_size - 1, patch_size - overlap)
    gridy = np.append(gridy, img_us_height - patch_size - 1)

    count = 0

    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            count += 1
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_patch = np.ravel(us_patch, order="F")
            # us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_mean = np.mean(us_patch)
            us_patch = np.ravel(us_patch, order="F") - us_mean
            # us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))
            us_norm = np.linalg.norm(us_patch)

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order="F")
            # feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))
            feat_norm = np.linalg.norm(feat_patch)

            if feat_norm > 1:
                # y = np.divide(feat_patch, feat_norm)
                y = feat_patch / feat_norm
            else:
                y = feat_patch

            # b = np.dot(np.multiply(Dl.T, -1), y)
            b = np.dot(-Dl.T, y)
            w = fss_yang(lmbd, Dl, b)

            hr_patch = np.dot(Dh, w)
            hr_patch = lin_scale(hr_patch, us_norm)

            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] += 1

    index = np.where(cnt_matrix < 1)[0]
    img_hr[index] = img_us[index]

    cnt_matrix[index] = 1
    img_hr = np.divide(img_hr, cnt_matrix)

    return img_hr
