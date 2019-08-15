from datasets import CTCSegmentationDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# coord_np = np.stack((np.mgrid[:imgshape[0], :imgshape[1]][0, :, :] / 16.,
#                   np.mgrid[:imgshape[0], :imgshape[1]][1, :, :] / 16.), 0)

# coord = torch.from_numpy(coord_np[None]).float().cuda()


# def average_over_mask(inp, mask):
#     return inp[mask].mean()


#     rx = np.random.randint(kernel_size, seg_numpy.shape[0] - kernel_size)
#     ry = np.random.randint(kernel_size, seg_numpy.shape[1] - kernel_size)

#     if seg_numpy[rx, ry] > 0:

#         sample_number += 1
#         window = seg_numpy[rx - kernel_size:rx + kernel_size, ry -
#                            kernel_size:ry + kernel_size]
#         size = (window == seg_numpy[rx, ry]).sum()

#         if seg_numpy[rx, ry] == 0:
#             mask = window == 0
#         else:
#             mask = window > 0

#         p_incluster = window == seg_numpy[rx, ry]

#         x = p_incluster * coord[0, 0, :2 * kernel_size, :2 * kernel_size]
#         y = p_incluster * coord[0, 1, :2 * kernel_size, :2 * kernel_size]

#         # shape loss
#         com_x = (x).sum() / (size + 0.0000001)
#         com_y = (y).sum() / (size + 0.0000001)
#         I = (p_incluster * ((coord[0, 0, :2 * kernel_size, :2 * kernel_size] - com_x.cuda())**2
#                             + (coord[0, 1, :2 * kernel_size, :2 * kernel_size] - com_y.cuda())**2)).sum()

#         avg_int = (p_incluster * img_c[0, rx - kernel_size:rx + kernel_size, ry - kernel_size:ry + kernel_size]).sum() / (size + 0.0000001)

#         yield rx, ry, size, I, mask, com_x, com_y, avg_int

if __name__ == '__main__':
    ds = CTCSegmentationDataset("/mnt/data1/swolf/CTC/DIC-C2DH-HeLa",
                                transforms="no")

    data = {}

    # Define a monopole, dipole, and (traceless) quadrupole by, respectively,
    # q_{\mathrm {tot} }\equiv \sum _{i=1}^{N}q_{i},\quad P_{\alpha }\equiv
    # \sum _{i=1}^{N}q_{i}r_{i\alpha },\quad {\hbox{and}}\quad Q_{\alpha \beta
    # }\equiv \sum _{i=1}^{N}q_{i}(3r_{i\alpha }r_{i\beta }-\delta _{\alpha
    # \beta }r_{i}^{2}),

    coord_np = None

    data["size"] = []
    data["avg_int"] = []
    data["size"] = []
    data["moment_of_inertia"] = []
    data["dp"] = []
    data["dpx"] = []
    data["dpy"] = []
    data["Q"] = []
    for i in range(2):
        for j in range(2):
            data[f"Q_{i}{j}"] = []


    for i, (img, seg) in enumerate(ds):

        if coord_np is None:
            coord_np = np.mgrid[:img.shape[1], :img.shape[2]][:, :, :]
            print(coord_np.shape)


        for s in np.unique(seg):
            if s == 0:
                continue

            mask = (seg[0] == s)

            # monopole moment aka size
            size = mask.sum()
            data["size"].append(size)

            data["avg_int"].append(img[0][mask].mean())

            com_x = coord_np[0][mask].mean()
            com_y = coord_np[1][mask].mean()

            center_coord = coord_np - np.array([[[com_x]], [[com_y]]])

            # data["avg_int"].append(img[0][mask].mean())

            # Dipole moments:
            dp = center_coord[:, mask].mean(axis=1)
            data["dp"].append(dp)
            data["dpx"].append(dp[0])
            data["dpy"].append(dp[1])

            # moment of inertia
            # $I_{P}=\sum _{i=1}^{N}m_{i}r_{i}^{2}.$

            moment_of_inertia = (center_coord ** 2).sum(axis=0)[mask].mean()
            data["moment_of_inertia"].append(moment_of_inertia)

            # quadrupole moments
            ndim = center_coord.shape[0]
            Q = np.zeros((ndim, ndim))
            for i in range(ndim):
                correction = (center_coord ** 2)[i, mask]
                for j in range(i, ndim):
                    q_per_pixel = 3 * (center_coord[i][mask] * center_coord[j][mask])
                    if i == j:
                        q_per_pixel -= correction
                    Q[i][j] = q_per_pixel.mean()
                    Q[j][i] = Q[i][j]

            for i in range(2):
                for j in range(2):
                    data[f"Q_{i}{j}"].append(Q[i][j])

            data["Q"].append(Q)

    df = pd.DataFrame(data=data)
    fig = df.plot.hist(subplots=True, rot=45, layout=(3,10)).get_figure()
    fig.savefig(f"analysis.pdf")
