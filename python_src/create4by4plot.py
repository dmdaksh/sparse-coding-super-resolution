from os import listdir , remove
from os.path import isdir
from PIL import Image


import matplotlib.pyplot as plt

addr = "data/results"

dirs = listdir(addr)


for dir in dirs:
    # check if dir is a directory
    if isdir(addr + "/" + dir):
        # remove plot.png if it exists
        try:
            remove(addr + "/" + dir + "/plot.png")
        except:
            pass
        img_hr = Image.open(addr + "/" + dir + "/3HR.png")
        img_lr = Image.open(addr + "/" + dir + "/LR.jpg")
        img_bc = Image.open(addr + "/" + dir + "/1bicubic.png")
        img_sr = Image.open(addr + "/" + dir + "/2SR.png")

        fig, axs = plt.subplots(2, 2)
        # make subplot aesthetics
        fig.subplots_adjust(hspace=0.5)
        fig.subplots_adjust(wspace=0.5)


        axs[0, 0].imshow(img_hr)
        axs[0, 0].set_title("HR")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(img_lr)
        axs[0, 1].set_title("LR")
        axs[0, 1].axis('off')

        axs[1, 0].imshow(img_bc)

        # get rms error from RMSE_bicubic.txt
        with open(addr + "/" + dir + "/RMSE_bicubic.txt") as f:
            rmse = f.read()

        rmse = round(float(rmse), 5)
        axs[1, 0].set_title("Bicubic\nRMSE: " + str(rmse))
        axs[1, 0].axis('off')

        axs[1, 1].imshow(img_sr)

        # get rms error from RMSE_SR.txt
        with open(addr + "/" + dir + "/RMSE_SR.txt") as f:
            rmse = f.read()

        rmse = round(float(rmse), 5)
        axs[1, 1].set_title("SR\nRMSE: " + str(rmse))
        axs[1, 1].axis('off')

        plt.savefig(addr + "/" + dir + "/plot.png", dpi=500)