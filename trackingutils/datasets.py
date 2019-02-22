from torch.utils.data import Dataset, DataLoader
from inferno.io.core import Zip, ZipReject
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.image import RandomFlip, RandomCrop
from inferno.io.transform.generic import Normalize
from inferno.io.transform.image import RandomRotate, ElasticTransform
from neurofire.transform.affinities import Segmentation2Affinities2D, Segmentation2Affinities
from inferno.io.volumetric import LazyHDF5VolumeLoader, HDF5VolumeLoader
from neurofire.criteria.loss_transforms import MaskTransitionToIgnoreLabel
from neurofire.criteria.loss_transforms import InvertTarget
from embeddingutils.transforms import Segmentation2AffinitiesWithPadding
from pathlib import Path
from .transforms import SliceTransform
import h5py
import torch
import numpy as np
import vigra


def get_transforms(transforms, dim, output_size, crop=True):
    if transforms == 'all':
        transform = Compose(RandomFlip(),
                            RandomRotate(),
                            ElasticTransform(alpha=2000., sigma=50., order=0))
        if crop:
            transform.add(RandomCrop(output_size))
        transform.add(AsTorchBatch(dim))

    elif transforms == 'all_affinities':

        offsets = [[-1, 0], [0, -1],
                   [-9, 0], [0, -9],
                   [-9, -9], [9, -9],
                   [-9, -4], [-4, -9], [4, -9], [9, -4],
                   [-27, 0], [0, -27]]

        transform = Compose(RandomFlip(),
                            RandomRotate(),
                            ElasticTransform(alpha=2000., sigma=50., order=0),
                            Segmentation2Affinities2D(offsets=offsets,
                                                      segmentation_to_binary=True,
                                                      apply_to=[1],
                                                      ignore_label=-1),
                            InvertTarget())
        if crop:
            transform.add(RandomCrop(output_size))
        transform.add(AsTorchBatch(dim))

    elif transforms == 'tracking_affinities':

        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-1, -4, -4], [-1, 4, 4],
                   [-1, -4, 4], [-1, 4, -4],
                   [-2, 0, 0],
                   [0, -9, 0], [0, 0, -9],
                   [0, -9, -9], [0, 9, -9],
                   [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                   [0, -27, 0], [0, 0, -27], [0, -81, 0], [0, 0, -81]]

        transform = Compose(RandomFlip(),
                            RandomRotate(),
                            Normalize(apply_to=[0]),
                            ElasticTransform(alpha=2000., sigma=50., order=0),
                            Segmentation2Affinities(offsets=offsets,
                                                    segmentation_to_binary=True,
                                                    apply_to=[1],
                                                    ignore_label=-1,
                                                    retain_segmentation=True))
        if crop:
            transform.add(RandomCrop(output_size))

        transform.add(AsTorchBatch(dim))

    elif transforms == '3D_tracking_affinities':

        offsets = [[-1, 0, 0, 0], [0, -1, 0, 0],
                   [0, 0, -1, 0], [0, 0, 0, -1],
                   [-1, 0, -4, -4], [-1, 0, 4, 4],
                   [-1, 0, -4, 4], [-1, 0, 4, -4],
                   [0, -1, -4, -4], [0, -1, 4, 4],
                   [0, -1, -4, 4], [0, -1, 4, -4],
                   [0, -2, 0, 0],
                   [0, 0, -9, 0], [0, 0, 0, -9],
                   [0, 0, -9, -9], [0, 0, 9, -9],
                   [0, 0, -9, -4], [0, 0, -4, -9],
                   [0, 0, 4, -9], [0, 0, 9, -4],
                   [0, 0, -27, 0], [0, 0, 0, -27]]

        transform = Compose(RandomFlip(),
                            RandomRotate(),
                            Normalize(apply_to=[0]),
                            ElasticTransform(alpha=2000., sigma=50., order=0),
                            Segmentation2AffinitiesWithPadding(offsets=offsets,
                                                               segmentation_to_binary=True,
                                                               apply_to=[1],
                                                               ignore_label=-1,
                                                               retain_segmentation=False),
                            SliceTransform(2, 1, apply_to=[1]))
        if crop:
            transform.add(RandomCrop(output_size))

        transform.add(AsTorchBatch(dim))

    elif transforms == 'minimal':
        transform = Compose()
        if crop:
            transform.add(AsTorchBatch(dim))
    elif transforms == "no":
        transform = None
    else:
        raise NotImplementedError()

    return transform


class CTCDataWrapper():
    """
    Provides a unified access method for the tiff files in the CTC
    Features a h5 file conversion and buffer layer

    Parameters:
    slice_z: set this parameter to True if the GT only provides only
             segmentation slices
    """

    def __init__(self, root_folder, use_buffer=True, normalize=True,
                 slice_z=False, load_3d=False, folders=None):
        # create list of available image files
        self.root_folder = Path(root_folder)
        self._buffer = {}
        self.load_3d = load_3d
        self.slice_z = slice_z
        self.use_buffer = use_buffer
        self.folders = folders
        self.normalize = normalize
        self.parse_root_folder()

    def parse_root_folder(self):

        self.image_files = {}
        self.segmentation_files = []
        self.fgbg_files = []
        self.tracking_files = []
        self.allimages = []
        self.link_files = {}

        for data_folder in sorted(self.root_folder.glob("[0-9][0-9]")):

            if self.folders is not None and data_folder.name not in self.folders:
                continue

            folder_number = int(data_folder.name)
            # print(data_folder.name, folder_number)
            self.image_files[folder_number] = {}
            # self.link_files[folder_number] = {}

            for image_file in sorted(data_folder.glob("t[0-9][0-9][0-9].tif")):
                t = int(image_file.name[1:4])
                self.image_files[folder_number][t] = {}
                self.image_files[folder_number][t]["raw_file"] = image_file
                self.allimages.append((folder_number, t))

                fg_file = data_folder.joinpath(f"t{t:03}_Probabilities.h5")
                if fg_file.exists():
                    self.image_files[folder_number][t]["fg_file"] = fg_file
                    self.fgbg_files.append((image_file, fg_file, (folder_number, t)))

                # check for GT folder
                gt_folder = self.root_folder.joinpath(f"{folder_number:02}_GT")
                if gt_folder.exists():
                    if not self.slice_z:
                        seg_file = gt_folder.joinpath("SEG", f"man_seg{t:03}.tif")
                        if seg_file.exists():
                            self.image_files[folder_number][t]["has_segmentation"] = True
                            self.image_files[folder_number][t]["segmentation_file"] = seg_file
                            self.segmentation_files.append((image_file, seg_file, (folder_number, t)))
                    else:
                        seg_slice_file = gt_folder.joinpath("SEG").glob(f"man_seg_{t:03}_[0-9][0-9][0-9].tif")
                        for seg_slice in seg_slice_file:
                            z = int(str(seg_slice).split("_")[-1][:3])
                            self.image_files[folder_number][t]["has_segmentation"] = True
                            self.image_files[folder_number][t]["segmentation_file"] = seg_slice
                            self.segmentation_files.append((image_file, seg_slice, (folder_number, t, z)))

                    fg_file = data_folder.joinpath(f"t{t:03}_Probabilities.h5")

                    if fg_file.exists():
                        self.image_files[folder_number][t]["fgbg_file"] = fg_file

                    tra_file = gt_folder.joinpath("TRA", f"man_track{t:03}.tif")
                    if tra_file.exists():
                        self.image_files[folder_number][t]["tracking_file"] = tra_file
                        self.tracking_files.append((image_file, tra_file, (folder_number, t)))

            link_file = gt_folder.joinpath("TRA", f"man_track.txt")
            if link_file.exists():
                self.link_files[folder_number] = link_file

    # TODO: test if this is truly depreciated
    # @property
    # def segmentation(self, to_numpy=True, use_buffer=True):
    #     for i in range(self.len_segmentation()):
    #         yield self.get_segmentation(i,
    #                                     to_numpy=to_numpy)

    def len_segmentation(self):
        return len(self.segmentation_files)

    def len_tracking(self):
        return len(self.tracking_files)

    def len_fgbg_files(self):
        return len(self.fgbg_files)

    def len_images(self):
        number_of_images = 0
        for folder in self.image_files:
            number_of_images += len(self.image_files[folder])

        return number_of_images

    def get_image(self, file_name, buffer_name=None,
                  normalize=False, h5=False, select_z=None):
        # create new buffer dictionary if it does not exist
        if self.use_buffer:
            if buffer_name not in self._buffer:
                self._buffer[buffer_name] = {}

            inbuffer = file_name in self._buffer[buffer_name]

        load_from_file = (self.use_buffer and not inbuffer) \
            or not self.use_buffer

        if load_from_file:
            if h5:
                with h5py.File(file_name, "r") as h5file:
                    img = h5file["exported_data"][..., 1]
            else:
                if self.load_3d:
                    img = vigra.impex.readVolume(str(file_name)).transpose(3, 2, 1, 0)
                else:
                    img = vigra.impex.readImage(str(file_name)).transpose(2, 1, 0)

            if select_z is not None:
                img = img[:, select_z]

            if normalize:
                # img /= 255.
                img -= img.mean()
                img /= img.std()

            if self.use_buffer:
                self._buffer[buffer_name][file_name] = img
        else:
            img = self._buffer[buffer_name][file_name]

        return img

    def get_segmentation(self, index, to_numpy=True):
        rf, sf, position = self.segmentation_files[index]
        if len(position) == 3 and self.slice_z:
            select_z = position[2]
            seg_select_z = 0
        else:
            select_z = None
            seg_select_z = None

        if not to_numpy:
            return rf, sf, position
        else:
            return self.get_image(rf,
                                  buffer_name="raw",
                                  normalize=self.normalize,
                                  select_z=select_z), \
                self.get_image(sf, buffer_name="seg",
                               select_z=seg_select_z)

    def get_tracking(self, index, to_numpy=True):
        rf, sf, _ = self.tracking_files[index]
        if not to_numpy:
            return rf, sf
        else:
            return self.get_image(rf, buffer_name="raw", normalize=True), \
                self.get_image(sf, buffer_name="tra")

    def get_allimages(self, index, to_numpy=True):

        # find folder matching to index
        folder, t = self.allimages[index]

        rf = self.image_files[folder][t]["raw_file"]
        sf = None
        if "has_segmentation" in self.image_files[folder][t]:
            sf = self.image_files[folder][t]["segmentation_file"]

        if not to_numpy:
            return rf, sf
        else:
            img = self.get_image(rf, buffer_name="raw", normalize=self.normalize)

            if sf is not None:
                seg = self.get_image(sf, buffer_name="seg")
            else:
                seg_shape = list(img.shape)
                seg_shape[0] = 1
                seg = np.zeros(seg_shape)

            return img, seg

    def get_fgbg_images(self, index, length=1):
        if length > 1:
            # shift index if the sequence would bridge two folders
            index = min(index, len(self.fgbg_files) - length)
            ref = self.fgbg_files[index][2]
            folders = [self.fgbg_files[index + l][2] == ref for l in range(length)]

            if np.sum(folders) < len(folders):
                index = index - len(folders) + np.sum(folders)

            images = [self.get_fgbg_images(index + l, length=1) for l in range(length)]
            return np.concatenate([img[0] for img in images]), \
                np.stack([img[1] for img in images])

        rf, fgfile, _ = self.fgbg_files[index]
        return self.get_image(rf, buffer_name="raw", normalize=True), \
            self.get_image(fgfile, buffer_name="fg", h5=True)


class CTCSegmentationDataset(Dataset):
    """
    Pytorch dataset for the cell tracking challenge
    loading segmentations only
    """

    def __init__(self, root_folder,
                 output_size=(128, 128),
                 transforms='all',
                 use_buffer=True,
                 slice_z=False,
                 folders=None,
                 normalize=True,
                 dim=2):

        self.data = CTCDataWrapper(root_folder,
                                   use_buffer=use_buffer,
                                   load_3d=(dim == 3),
                                   slice_z=slice_z,
                                   normalize=normalize,
                                   folders=folders)

        self.transform = get_transforms(transforms, dim, output_size, False)

        self.output_size = output_size

    def __len__(self):
        return self.data.len_segmentation()

    def __getitem__(self, idx):
        img, gt = self.data.get_segmentation(idx)
        img, gt = img.astype(np.float32), gt.astype(np.float32)
        if self.transform is not None:
            img, gt = self.transform(img, gt)

        return img[0], gt[0]


class CTCDataset(CTCSegmentationDataset):

    def __len__(self):
        return self.data.len_images()

    def __getitem__(self, idx):
        img, gt = self.data.get_allimages(idx)
        img = img.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        return img


class CTCSemisupervisedSegmentationDataset(CTCDataset):
    """
    Pytorch dataset for the cell tracking challenge
    loading segmentations only
    """

    def __init__(self, *args, **kwargs):
        super(CTCSemisupervisedSegmentationDataset, self).__init__(*args, **kwargs)
        self.compute_object_features()

    def compute_object_features(self):

        coord_np = None
        # accumulate the shape features in the data dict
        shape_feats = {}
        shape_feats["size"] = []
        shape_feats["avg_int"] = []
        shape_feats["size"] = []
        shape_feats["moment_of_inertia"] = []
        shape_feats["dp"] = []
        shape_feats["dpx"] = []
        shape_feats["dpy"] = []

        for i in range(2):
            for j in range(i, 2):
                shape_feats[f"Q_{i}{j}"] = []

        for i in range(self.data.len_segmentation()):

            img, seg, _ = self.data.get_segmentation(i)

            if coord_np is None:
                coord_np = np.mgrid[:img.shape[1], :img.shape[2]][:, :, :]

            for s in np.unique(seg):
                if s == 0:
                    continue

                mask = (seg[0] == s)

                # monopole moment aka size
                size = mask.sum()
                shape_feats["size"].append(size)

                shape_feats["avg_int"].append(img[0][mask].mean())

                com_x = coord_np[0][mask].mean()
                com_y = coord_np[1][mask].mean()

                center_coord = coord_np - np.array([[[com_x]], [[com_y]]])

                # Dipole moments:
                dp = center_coord[:, mask].mean(axis=1)
                shape_feats["dp"].append(dp)
                shape_feats["dpx"].append(dp[0])
                shape_feats["dpy"].append(dp[1])

                moment_of_inertia = (center_coord ** 2).sum(axis=0)[mask].mean()
                shape_feats["moment_of_inertia"].append(moment_of_inertia)
                print(moment_of_inertia)

                # quadrupole moments
                ndim = center_coord.shape[0]
                for i in range(ndim):
                    correction = (center_coord ** 2)[i, mask]
                    for j in range(i, ndim):
                        q_per_pixel = 3 * (center_coord[i][mask] * center_coord[j][mask])
                        if i == j:
                            q_per_pixel -= correction
                        shape_feats[f"Q_{i}{j}"] = q_per_pixel.mean()
            print(shape_feats)

        self.stds = np.zeros((len(shape_feats), ))
        self.means = np.zeros((len(shape_feats), ))

        for i, key in enumerate(shape_feats):
            self.means[i] = np.mean(shape_feats[key])
            self.stds[i] = np.std(shape_feats[key])

        del shape_feats


class CTCFgBgDataset(CTCSegmentationDataset):
    """
    Pytorch dataset for the cell tracking challenge
    loading segmentations only
    """

    def __init__(self, *args, npairs=1, **kwargs):
        self.npairs = npairs
        # if npairs > 1:
        #     for folder_number in self.image_files.keys():
        #         self.indices.extend(list(image_files[t][:-npairs]))
        super(CTCFgBgDataset, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.data.len_images()

    def __getitem__(self, idx):
        img, fg = self.data.get_fgbg_images(idx, length=self.npairs)
        img, fg = img.astype(np.float32), fg.astype(np.float32)
        print(img.shape, fg.shape)
        if self.transform is not None:
            img, fg = self.transform(img, fg)
        if self.npairs > 1:
            return img, np.stack((img, fg), axis=1)
        else:
            return img, np.concatenate((img, fg), axis=0)


class CTCSyntheticDataset(Dataset):

    def __init__(self,
                 root_folder,
                 output_size=(128, 128),
                 dim=2,
                 transforms='all',
                 **kwargs):

        self.transform = get_transforms(transforms, dim, output_size)
        self.output_size = output_size

    def __len__(self):
        # return a random number since we are generating new images anyway
        return 1000

    def __getitem__(self, idx):

        img = self.generate_image()
        img, gt, _ = self.data.get_segmentation(idx)
        img, gt = img.astype(np.float32), gt.astype(np.float32)

        if self.transform is not None:
            img, gt = self.transform(img, gt)

        return img, gt


class RejectFewInstances(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, fetched):
        ratio = ((fetched > 0).sum() / fetched.size)
        return ratio < self.threshold


# TODO: move this into inferno???
class MergeDataset(Dataset):

    def __init__(self, *datasets):
        self.datasets = tuple(*datasets)

    def __len__(self):
        # return a random number since we are generating new images anyway
        return sum(len(d) for d in self.datasets)

    def index_to_ds_subindex(self, index):
        d_idx = 0
        while (index >= len(self.datasets[d_idx])):
            index -= len(self.datasets[d_idx])
            d_idx += 1
        return d_idx, index

    def __getitem__(self, idx):
        d_idx, sub_idx = self.index_to_ds_subindex(idx)
        return self.datasets[d_idx].__getitem__(sub_idx)


class CTCAffinityDataset(ZipReject):

    def __init__(self, root_folder, h5file, shape, dim, use_time_as_channels=False, stride=None,
                 rejection_threshold=0.001, folders=None, transforms='all_affinities'):

        data_filename = str(Path(root_folder).joinpath(h5file))

        if folders is None:
            with h5py.File(data_filename, "r") as h5file:
                folders = [k for k in h5file.keys()]

        if stride is None:
            stride = [max(s // 2, 1) for s in shape]

        raw_ds = MergeDataset(
            LazyHDF5VolumeLoader(data_filename, f'{f}/raw',
                                 window_size=shape,
                                 stride=stride,
                                 padding=None,
                                 padding_mode='constant')
            for f in folders)

        segmentatoin_ds = MergeDataset(
            LazyHDF5VolumeLoader(data_filename, f'{f}/tracklet_seg',
                                 window_size=shape,
                                 stride=stride,
                                 padding=None,
                                 padding_mode='constant')
            for f in folders)

        batch_dim = dim
        if use_time_as_channels:
            batch_dim -= 1

        self.use_time_as_channels = use_time_as_channels

        self.transform = get_transforms(transforms, batch_dim, None, crop=False)

        super(CTCAffinityDataset, self).__init__(raw_ds, segmentatoin_ds,
                                                 rejection_dataset_indices=1,
                                                 rejection_criterion=RejectFewInstances(rejection_threshold),
                                                 sync=True)

    def __getitem__(self, idx):

        img, gt = super(CTCAffinityDataset, self).__getitem__(idx)

        if self.transform is not None:
            img, gt = self.transform(img, gt.astype(np.int64))

        return img.type(torch.FloatTensor), gt.type(torch.FloatTensor)


if __name__ == '__main__':

    # ds = CTCSegmentationDataset("/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC",
    #                             dim=3,
    #                             slice_z=True,
    #                             output_size=(512, 512))

    # print("ds size ", len(ds))
    # for i, (img, seg) in enumerate(ds):
    #     print(img.shape, seg.shape)
    # shape = [8, 512, 512]

    HDF5VolumeLoader("/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC/data.h5", '02/raw',
                     window_size=[4, 13, 256, 256],
                     stride=[1, 1, 128, 128],
                     padding=None,
                     padding_mode='constant')

    # ds = CTCAffinityDataset("/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC",
    #                         shape=[4, 13, 256, 256],
    #                         dim=4,
    #                         h5file="data.h5",
    #                         folders=['02'],
    #                         transforms='tracking_affinities',
    #                         use_time_as_channels=True)

    # print("start")
    # for epoch in range(10):
    #     for i, img in enumerate(ds):
    #         print(img)
    #         with h5py.File("debug.h5", "w") as h5file:
    #             h5file.create_dataset("img0", data=img[0])
    #             h5file.create_dataset("img1", data=img[1])
