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


def already_processed(tracking_data, k0, k1):
    for k in tracking_data["size"]:
        if k[0] == k0 and k[1] == k1:
            return True
    return False

if __name__ == '__main__':
    root_folder = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC"
    output_file = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC/tracking_approx.h5"
    ndim = 3

    # root_folder = "/mnt/data1/swolf/CTC/PhC-C2DL-PSC"
    # ndim = 2

    compute_features = True
    if len(sys.argv) > 1:
        compute_features = sys.argv[1] == "compute"

    dict_file = Path(root_folder).joinpath('tracking_data.pickle')

    if compute_features:
        ds = CTCDataWrapper(root_folder, load_3d=ndim == 3)

        if dict_file.exists():
            with open(dict_file, 'rb') as handle:
                tracking_data = pickle.load(handle)
        else:
            tracking_data = {}
            tracking_data["size"] = {}
            tracking_data["avg_int"] = {}
            tracking_data["dp"] = {}
            if ndim == 3:
                tracking_data["dp_2"] = {}
            tracking_data["dp_1"] = {}
            tracking_data["dp_0"] = {}
            if ndim == 3:
                tracking_data["com_2"] = {}
            tracking_data["com_1"] = {}
            tracking_data["com_0"] = {}
            tracking_data["moment_of_inertia"] = {}

            tracking_data["maxdist"] = {}

            for i in range(ndim):
                for j in range(ndim):
                    tracking_data[f"Q_{i}{j}"] = {}

        all_labels = {}
        for index in tqdm(range(ds.len_tracking())):

            img, y = ds.get_tracking(index)
            folder_name = str(ds.get_tracking(index, to_numpy=False)[0]).split("/")[-2]

            folder_number, t = ds.tracking_files[index][2]
            if folder_name not in all_labels:
                all_labels[folder_name] = []

            if already_processed(tracking_data, folder_number, t):
                continue

            prob_file = ds.image_files[folder_number][t]["fg_file"]
            with h5py.File(prob_file, "r") as h5file:
                if ndim == 3:
                    fg_porb = gaussian_filter(h5file["exported_data"][..., 1], [0.1, 2.5, 2.5])
                else:
                    fg_porb = gaussian_filter(h5file["exported_data"][..., 1], [1., 1.])
                    # imsave(Path(root_folder).joinpath('fg.debug.png'), fg_porb)

            fg = fg_porb > 0.7
            fg[:3] = 0
            labels = watershed(1 - fg_porb, y[0], mask=fg_porb > 0.7)

            ignore_label = fg.copy()
            ignore_label[labels > 0] = 0
            labels[ignore_label] = -1
            labels[:3] = -1

            all_labels[folder_name].append(labels)
            imgshape = labels.shape

            if ndim == 3:
                coord_np = np.mgrid[:imgshape[0], :imgshape[1], :imgshape[2]] / 16.
            else:
                coord_np = np.mgrid[:imgshape[0], :imgshape[1]] / 16.

            # for idx in np.unique(labels):

            #     print(index, idx)
            #     # skip background
            #     if idx == 0:
            #         continue

            #     mask = (labels == idx)

            #     # monopole moment aka size
            #     size = mask.sum()
            #     tracking_data["size"][(folder_number, t, idx)] = size
            #     tracking_data["avg_int"][(folder_number, t, idx)] = img[0][mask].mean()

            #     if ndim == 3:
            #         com_2 = coord_np[2][mask].mean()
            #         tracking_data["com_2"][(folder_number, t, idx)] = com_2
            #     com_1 = coord_np[1][mask].mean()
            #     tracking_data["com_1"][(folder_number, t, idx)] = com_1
            #     com_0 = coord_np[0][mask].mean()
            #     tracking_data["com_0"][(folder_number, t, idx)] = com_0

            #     if ndim == 3:
            #         center_coord = coord_np[:, mask] - np.array([[com_0], [com_1], [com_2]])
            #     else:
            #         center_coord = coord_np[:, mask] - np.array([[com_0], [com_1]])

            #     distance = (torch.from_numpy(center_coord)**2).sum(0)
            #     maxdist = torch.logsumexp(distance, 0)
            #     tracking_data["maxdist"][(folder_number, t, idx)] = maxdist.numpy()

            #     # tracking_data["avg_int"][(folder_number, t, idx)].append(img[0][mask].mean())

            #     # Dipole moments:
            #     dp = center_coord.mean(axis=1)
            #     tracking_data["dp"][(folder_number, t, idx)] = dp
            #     if ndim == 3:
            #         tracking_data["dp_2"][(folder_number, t, idx)] = dp[2]
            #     tracking_data["dp_1"][(folder_number, t, idx)] = dp[1]
            #     tracking_data["dp_0"][(folder_number, t, idx)] = dp[0]

            #     # moment of inertia
            #     # $I_{P}=\sum _{i=1}^{N}m_{i}r_{i}^{2}.$

            #     moment_of_inertia = (center_coord ** 2).sum(axis=0).mean()
            #     tracking_data["moment_of_inertia"][(folder_number, t, idx)] = moment_of_inertia

            #     # quadrupole moments
            #     ndim = center_coord.shape[0]
            #     Q = np.zeros((ndim, ndim))
            #     for i in range(ndim):
            #         correction = (center_coord ** 2)[i]
            #         for j in range(i, ndim):
            #             q_per_pixel = 3 * (center_coord[i] * center_coord[j])
            #             if i == j:
            #                 q_per_pixel -= correction
            #             Q[i][j] = q_per_pixel.mean()
            #             Q[j][i] = Q[i][j]

            #     for i in range(ndim):
            #         for j in range(ndim):
            #             tracking_data[f"Q_{i}{j}"][(folder_number, t, idx)] = Q[i][j]

            with open(dict_file, 'wb') as handle:
                pickle.dump(tracking_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if index % 10 == 0:
                print("writing label file temporarily")
                with h5py.File(output_file, "w") as tracking_outfile:
                    for fdn in all_labels:
                        tracking_outfile.create_dataset(f"{fdn}/tracklet_seg",
                                                        data=np.stack(all_labels[fdn]),
                                                        compression='gzip')

        print("writing label file")
        with h5py.File(output_file, "w") as tracking_outfile:
            for fdn in all_labels:
                tracking_outfile.create_dataset(f"{fdn}/tracklet_seg",
                                                data=np.stack(all_labels[fdn]),
                                                compression='gzip')

    with open(dict_file, 'rb') as handle:
        tracking_data = pickle.load(handle)

    tracklets = set((folder_number, idx) for folder_number, t, idx in tracking_data["size"])

    with h5py.File(root_folder + "/model_parameters.h5", "w") as h5file:

        do_not_consider = ["dp", "com_2", "com_1", "com_0", "dp_0", "dp_1", "dp_2", "Q_10", "Q_20", "Q_21"]
        used_keys = [k for k in tracking_data.keys() if k not in do_not_consider]
        # for k in [tracking_data ]:
        #     print(k, np.array(list(tracking_data[k].values())).shape)
        # #     print(k, np.mean(list(tracking_data[k].values())), np.std(list(tracking_data[k].values())))

        X = np.stack([np.array(list(tracking_data[k].values())) for k in used_keys], axis=0)
        # cov = np.linalg.inv(np.cov(X))
        # print((2 * np.pi)**len(used_keys) * np.linalg.det(cov))
        # cov /= np.sqrt((2 * np.pi)**len(used_keys) * np.linalg.det(cov))

        from sklearn.covariance import EmpiricalCovariance, MinCovDet
        robust_cov = MinCovDet(support_fraction=0.7).fit(X.T)

        lw_cov_, _ = ledoit_wolf(X.T)
        lw_prec_ = linalg.inv(lw_cov_)

        h5file.create_dataset("self_covariance", data=lw_prec_)
        h5file["self_covariance"].attrs["keys"] = ";".join(used_keys)
        h5file.create_dataset("self_mean", data=np.mean(X, axis=1))
        h5file["self_mean"].attrs["keys"] = ";".join(used_keys)
        h5file.create_dataset("self_std", data=np.std(X, axis=1))
        h5file["self_std"].attrs["keys"] = ";".join(used_keys)

        do_not_consider = ["dp", "dp_0", "dp_1", "dp_2", "Q_10", "Q_20", "Q_21"]
        used_keys = [k for k in tracking_data.keys() if k not in do_not_consider]
        print(used_keys)
        pairs = []

        for fn, cell_id in tracklets:
            pos = list((folder_number, t, idx)
                       for folder_number, t, idx in tracking_data["size"] if (idx == cell_id and folder_number == fn))

            size = np.array([tracking_data["size"][p] for p in pos])
            mask = size < 4000

            X = np.stack([np.array([tracking_data[k][p] for p in pos]) for k in used_keys], axis=0)[:, mask]

            # pairs.append(np.concatenate((X[:, :-1], X[:, 1:]), axis=0))
            pairs.append(np.array(X[:, 1:] - X[:, :-1]))

            # stuff = medfilt([tracking_data[k][p] for p in pos], kernel_size=9)
            stuff = np.array([tracking_data["maxdist"][p] for p in pos])

            if len(stuff) > 2:
                time = np.array([p[1] for p in pos])

                plt.plot(time[size < 4000], stuff[size < 4000])

        plt.savefig(Path(root_folder).joinpath("size.png"))
        plt.clf()
        plt.close()

        pairs = np.concatenate(pairs, axis=1)
        # cov = np.linalg.inv(np.cov(pairs))
        # cov /= np.sqrt((2 * np.pi)**len(used_keys) * np.linalg.det(cov))

        p_lw_cov_, _ = ledoit_wolf(pairs.T)
        p_lw_prec_ = linalg.inv(lw_cov_)

        robust_cov = MinCovDet(support_fraction=0.7).fit(pairs.T)
        h5file.create_dataset("pair_covariance", data=p_lw_prec_)
        h5file["pair_covariance"].attrs["keys"] = ";".join(used_keys)
        # h5file.create_dataset("pair_mean", data=robust_cov.location_)
        # h5file["pair_mean"].attrs["keys"] = ";".join(used_keys)

        h5file.create_dataset("pair_mean", data=np.mean(X, axis=1))
        h5file["pair_mean"].attrs["keys"] = ";".join(used_keys)
        h5file.create_dataset("pair_std", data=np.std(X, axis=1))
        h5file["pair_std"].attrs["keys"] = ";".join(used_keys)

        # imsave("paircov.png", np.log(np.abs(np.cov(pairs))))

        # pairs

    # coord_np = None
    # data["size"] = []
    # data["avg_int"] = []
    # data["size"] = []
    # data["moment_of_inertia"] = []
    # data["dp"] = []
    # data["dpx"] = []
    # data["dpy"] = []
    # data["Q"] = []
    # for i in range(2):
    #     for j in range(2):
    #         data[f"Q_{i}{j}"] = []

    # for i, (img, seg) in enumerate(ds):

    #     if coord_np is None:
    #         coord_np = np.mgrid[:img.shape[1], :img.shape[2]][:, :, :]
    #         print(coord_np.shape)

    #     for s in np.unique(seg):
    #         if s == 0:
    #             continue

    #         mask = (seg[0] == s)

    #         # monopole moment aka size
    #         size = mask.sum()
    #         data["size"].append(size)

    #         data["avg_int"].append(img[0][mask].mean())

    #         com_x = coord_np[0][mask].mean()
    #         com_y = coord_np[1][mask].mean()

    #         center_coord = coord_np - np.array([[[com_x]], [[com_y]]])

    #         # data["avg_int"].append(img[0][mask].mean())

    #         # Dipole moments:
    #         dp = center_coord[:, mask].mean(axis=1)
    #         data["dp"].append(dp)
    #         data["dpx"].append(dp[0])
    #         data["dpy"].append(dp[1])

    #         # moment of inertia
    #         # $I_{P}=\sum _{i=1}^{N}m_{i}r_{i}^{2}.$

    #         moment_of_inertia = (center_coord ** 2).sum(axis=0)[mask].mean()
    #         data["moment_of_inertia"].append(moment_of_inertia)

    #         # quadrupole moments
    #         ndim = center_coord.shape[0]
    #         Q = np.zeros((ndim, ndim))
    #         for i in range(ndim):
    #             correction = (center_coord ** 2)[i, mask]
    #             for j in range(i, ndim):
    #                 q_per_pixel = 3 * (center_coord[i][mask] * center_coord[j][mask])
    #                 if i == j:
    #                     q_per_pixel -= correction
    #                 Q[i][j] = q_per_pixel.mean()
    #                 Q[j][i] = Q[i][j]

    #         for i in range(2):
    #             for j in range(2):
    #                 data[f"Q_{i}{j}"].append(Q[i][j])

    #         data["Q"].append(Q)

    # df = pd.DataFrame(data=data)
    # fig = df.plot.hist(subplots=True, rot=45, layout=(3,10)).get_figure()
    # fig.savefig(f"analysis.pdf")
