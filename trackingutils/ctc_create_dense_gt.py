from datasets import CTCSegmentationDataset, CTCDataWrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
from pathlib import Path
from scipy.ndimage.filters import gaussian_filter
import sys
from scipy.misc import imsave
from scipy import linalg
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from tqdm import tqdm

from scipy.signal import medfilt
import torch
from skimage.morphology import watershed
from scipy.ndimage.morphology import distance_transform_edt


if __name__ == '__main__':
    root_folder = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC"
    output_file = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC/tracking_approx.h5"
    ndim = 3

    crops = {"02": [slice(None), slice(150, 1500), slice(100, 1500)],
             "01": [slice(None), slice(500, 2100), slice(140, 1150)]}

    # root_folder = "/mnt/data1/swolf/CTC/PhC-C2DL-PSC"
    # ndim = 2

    ds = CTCDataWrapper(root_folder, load_3d=ndim == 3, normalize=False)

    all_labels = {}
    all_images = {}
    for index in tqdm(range(ds.len_tracking())):

        folder_name = str(ds.get_tracking(index, to_numpy=False)[0]).split("/")[-2]

        img, y = ds.get_tracking(index)

        folder_number, t = ds.tracking_files[index][2]
        if folder_name not in all_labels:
            all_labels[folder_name] = []
        if folder_name not in all_images:
            all_images[folder_name] = []


        prob_file = ds.image_files[folder_number][t]["fg_file"]
        with h5py.File(prob_file, "r") as h5file:
            if ndim == 3:
                fg_porb = gaussian_filter(h5file["exported_data"][..., 1], [0.1, 2.5, 2.5])
            else:
                fg_porb = gaussian_filter(h5file["exported_data"][..., 1], [1., 1.])
                # imsave(Path(root_folder).joinpath('fg.debug.png'), fg_porb)

        fg = fg_porb > 0.5
        if folder_number == 1:
            fg[:3] = 0
        fg[y[0] > 0] = 1
        



        # dist_to_seeds = distance_transform_edt(y[0] < 0.1)
        # potential_cell = (dist_to_seeds < 8)



        # other_seeds = distance_transform_edt(fg) < 7
        # other_seeds[potential_cell] = 0

        # distance_from_fake_seeds = distance_transform_edt(~other_seeds)

        # # distance_from_bg = distance_transform_edt(fg)

        # # might_be_seed = distance_from_bg < 7.


        # all_labels[folder_name].append(y[0][crops[folder_name]].astype(np.float32))
        # all_labels[folder_name].append(dist_to_seeds[crops[folder_name]].astype(np.float32))
        # all_labels[folder_name].append(distance_from_fake_seeds[crops[folder_name]].astype(np.float32))
        # # dist -= distance_transform_edt(fg)
        # # dist -= dist.min()
        # # dist /= dist.max()

        # # idea: 
        # # esitmate seeds based on distance from boundary
        # # measure distance from theses seeds
        # # use WS only in places where dist from true seeds < dist from fake seeds

        # max_dist_fg = fg.copy()
        # # max_dist_fg.fill(False)
        # msk = distance_from_fake_seeds < dist_to_seeds
        # max_dist_fg[msk] = 0

        # all_labels[folder_name].append((distance_from_fake_seeds - dist_to_seeds)[crops[folder_name]].astype(np.float32))
        # all_labels[folder_name].append(msk[crops[folder_name]].astype(np.float32))
        # all_labels[folder_name].append(max_dist_fg[crops[folder_name]].astype(np.float32))
        # dist_to_seeds += 0.001 * np.random.rand(*dist_to_seeds.shape)
        

        dist_to_seeds = distance_transform_edt(y[0] < 0.1)
        potential_cell = fg.copy()
        potential_cell[dist_to_seeds > 13] = 0

        dist_to_fg = distance_transform_edt(fg)


        labels = watershed(-dist_to_fg+dist_to_seeds, y[0], mask=potential_cell)

        ignore_label = fg.copy()
        ignore_label[labels > 0] = 0
        labels[ignore_label] = -1
        if folder_number == 1:
            labels[:3] = -1

        # all_labels[folder_name].append(labels[crops[folder_name]])
        # all_images[folder_name].append(img[0][crops[folder_name]])
        # all_images[folder_name].append(dist_to_seeds)
        imgshape = labels.shape

        output_file = f"/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC/tracking_approx_{folder_number:02}_{t:03}.h5"

        with h5py.File(output_file, "w") as tracking_outfile:
            tracking_outfile.create_dataset(f"{folder_number}/tracklet_seg",
                                            data=labels[crops[folder_name]],
                                            compression='gzip')
            tracking_outfile.create_dataset(f"{folder_number}/img",
                                            data=img[0][crops[folder_name]])
    # print("writing label file")
    # with h5py.File(output_file, "w") as tracking_outfile:
    #     for fdn in all_labels:
    #         tracking_outfile.create_dataset(f"{fdn}/tracklet_seg",
    #                                         data=np.stack(all_labels[fdn]),
    #                                         compression='gzip')
    #         tracking_outfile.create_dataset(f"{fdn}/img",
    #                                         data=np.stack(all_images[fdn]))