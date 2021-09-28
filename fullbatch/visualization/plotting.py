"""Plotting helpers."""

import os
import matplotlib.pyplot as plt

from .database import load_surface_from_lmdb

# import matplotlib.font_manager as font_manager
#
# font_dirs = ['/nfshomes/pepope/share/fonts', '/usr/share/fonts/']
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# font_list = font_manager.createFontList(font_files)
# font_manager.fontManager.ttflist.extend(font_list)


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman Bold'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 4


def plot_1d_loss_err_row(base_data_dir, db_names, display_names, xcoords, positions, figsize=None,
                         xmin=-1.0, xmax=1.0, loss_max=5, log=False,):
    """
    1D plotting routines

    FORKED FROM
    * https://github.com/tomgoldstein/loss-landscape/blob/master/plot_1D.py
    """
    ncols = len(db_names)
    nrows = 1
    if not figsize:
        figsize = (5 * ncols, 5 * nrows)
    f, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = [axes] if ncols == 1 else axes
    for i, (ax1, db_name) in enumerate(zip(axes, db_names)):

        file_path = os.path.join(base_data_dir, db_name)
        landscape = load_surface_from_lmdb(file_path, positions)

        xmin = xmin if xmin != 'min' else min(x)
        xmax = xmax if xmax != 'max' else max(x)

        # loss and accuracy map
        ax2 = ax1.twinx()
        if log:
            tr_loss, = ax1.semilogy(xcoords, landscape['train_loss'], 'b-', label='Training loss', linewidth=1)
        else:
            tr_loss, = ax1.plot(xcoords, landscape['train_loss'], 'b-', label='Training loss', linewidth=1)
        tr_acc, = ax2.plot(xcoords, landscape['train_acc'] * 100, 'r-', label='Training accuracy', linewidth=1)

        plt.xlim(xmin, xmax)
        if i == 0:
            ax1.set_ylabel('Loss', color='b', fontsize='xx-large')
        ax1.tick_params('y', colors='b', labelsize='x-large')
        ax1.tick_params('x', labelsize='x-large')
        ax1.set_ylim(0, loss_max)
        if i == ncols - 1:
            ax2.set_ylabel('Accuracy', color='r', fontsize='xx-large')
        ax2.tick_params('y', colors='r', labelsize='x-large')
        ax2.set_ylim(0, 100)


        display_name = display_names[i]
        ax1.set_title(display_name, fontsize='x-large', y=1.03)
    f.tight_layout()
    return f, landscape
