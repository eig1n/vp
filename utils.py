import os
import logging
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

def print_log(message):
    print(message)
    logging.info(message)

def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


from typing import Union

import cv2
import imageio
import numpy as np

import matplotlib.pyplot as plt

def show_video_line(data, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png', out_path=None, use_rgb=False):
    """generate images with a video sequence"""
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.25 * ncols, 3))
    plt.subplots_adjust(wspace=0.01, hspace=0)

    if len(data.shape) > 3:
        data = data.swapaxes(1,2).swapaxes(2,3)

    images = []
    if ncols == 1:
        if use_rgb:
            im = axes.imshow(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
        else:
            im = axes.imshow(data[0], cmap=cmap, norm=norm)
        images.append(im)
        axes.axis('off')
        im.set_clim(vmin, vmax)
    else:
        for t, ax in enumerate(axes.flat):
            if use_rgb:
                im = ax.imshow(cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB), cmap='gray')
            else:
                im = ax.imshow(data[t], cmap=cmap, norm=norm)
            images.append(im)
            ax.axis('off')
            im.set_clim(vmin, vmax)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7])
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    plt.show()
    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()