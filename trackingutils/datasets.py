from torch.utils.data import Dataset, DataLoader
from inferno.io.transform import Compose
from inferno.io.transform.generic import AsTorchBatch
from inferno.io.transform.image import RandomFlip, RandomCrop
from inferno.io.transform.image import RandomRotate, ElasticTransform
from pathlib import Path
import h5py
import numpy as np
import vigra


class CTCDataWrapper():
    """
    Provides a unified access method for the tiff files in the CTC
    Features a h5 file conversion and buffer layer
    """

    def __init__(self, root_folder, use_buffer=True, load_3d=False):
        # create list of available image files
        self.root_folder = Path(root_folder)
        self.parse_root_folder()
        self._buffer = {}
        self.load_3d = load_3d
        self.use_buffer = use_buffer

    def parse_root_folder(self):

        self.image_files = {}
        self.segmentation_files = []
        self.tracking_files = []
        self.link_files = {}

        for data_folder in sorted(self.root_folder.glob("[0-9][0-9]")):
            folder_number = int(data_folder.name)
            # print(data_folder.name, folder_number)
            self.image_files[folder_number] = {}
            # self.link_files[folder_number] = {}

            for image_file in sorted(data_folder.glob("t[0-9][0-9][0-9].tif")):
                t = int(image_file.name[1:4])
                self.image_files[folder_number][t] = {}
                self.image_files[folder_number][t]["raw_file"] = image_file

                fg_file = data_folder.joinpath(f"t{t:03}_Probabilities.h5")
                if fg_file.exists():
                    self.image_files[folder_number][t]["fg_file"] = fg_file

                # check for GT folder
                gt_folder = self.root_folder.joinpath(f"{folder_number:02}_GT")
                if gt_folder.exists():
                    seg_file = gt_folder.joinpath("SEG", f"man_seg{t:03}.tif")
                    if seg_file.exists():
                        self.image_files[folder_number][t]["has_segmentation"] = True
                        self.image_files[folder_number][t]["segmentation_file"] = seg_file
                        self.segmentation_files.append((image_file, seg_file, (folder_number, t)))

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

    @property
    def segmentation(self, to_numpy=True, use_buffer=True):
        for i in range(self.len_segmentation()):
            yield self.get_segmentation(i,
                                        to_numpy=to_numpy)

    def len_segmentation(self):
        return len(self.segmentation_files)

    def len_tracking(self):
        return len(self.tracking_files)

    def len_images(self):
        number_of_images = 0
        for folder in self.image_files:
            number_of_images += len(self.image_files[folder])

        return number_of_images

    def get_image(self, file_name, buffer_name=None, normalize=False):
        # create new buffer dictionary if it does not exist
        if self.use_buffer:
            if buffer_name not in self._buffer:
                self._buffer[buffer_name] = {}

        inbuffer = file_name in self._buffer[buffer_name]
        load_from_file = (self.use_buffer and not inbuffer) \
            or not self.use_buffer

        if load_from_file:
            if self.load_3d:
                img = vigra.impex.readVolume(str(file_name)).transpose(3, 2, 1, 0)
            else:
                img = vigra.impex.readImage(str(file_name)).transpose(2, 1, 0)
            if normalize:
                img -= img.mean()
                img /= img.std()

            if self.use_buffer:
                self._buffer[buffer_name][file_name] = img
        else:
            img = self._buffer[buffer_name][file_name]

        return img

    def get_segmentation(self, index, to_numpy=True):
        rf, sf, _ = self.segmentation_files[index]
        if not to_numpy:
            return rf, sf
        else:
            return self.get_image(rf, buffer_name="raw", normalize=True), \
                self.get_image(sf, buffer_name="seg")

    def get_tracking(self, index, to_numpy=True):
        rf, sf, _ = self.tracking_files[index]
        if not to_numpy:
            return rf, sf
        else:
            return self.get_image(rf, buffer_name="raw", normalize=True), \
                self.get_image(sf, buffer_name="tra")

    def get_allimages(self, index, to_numpy=True):

        # find folder matching to index
        remaining_index = int(index)
        for folder in self.image_files:
            if remaining_index > len(self.image_files[folder]):
                remaining_index -= len(self.image_files[folder])
            else:
                break

        rf = self.image_files[folder][remaining_index]["raw_file"]
        sf = None
        if self.image_files[folder][remaining_index]["has_segmentation"]:
            sf = self.image_files[folder][remaining_index]["segmentation_file"]

        if not to_numpy:
            return rf, sf
        else:
            img = self.get_image(rf, buffer_name="raw", normalize=True)

            if sf is not None:
                seg = self.get_image(sf, buffer_name="seg")
            else:
                seg_shape = list(img.shape)
                seg_shape[0] = 1
                seg = np.zeros(seg_shape)

            return img, seg


class CTCSegmentationDataset(Dataset):
    """
    Pytorch dataset for the cell tracking challenge
    loading segmentations only
    """

    def __init__(self, root_folder,
                 output_size=(128, 128),
                 transforms='all',
                 use_buffer=True):

        self.data = CTCDataWrapper(root_folder,
                                   use_buffer=use_buffer)

        if transforms == 'all':
            self.transform = Compose(RandomFlip(),
                                     RandomRotate(),
                                     ElasticTransform(alpha=2000., sigma=50., order=0),
                                     RandomCrop(output_size),
                                     AsTorchBatch(2))
        elif transforms == 'minimal':
            self.transform = Compose(RandomFlip(),
                                     RandomRotate(),
                                     RandomCrop(output_size),
                                     AsTorchBatch(2))
        elif transforms == 'no':
            self.transform = None
        else:
            self.transform = Compose(RandomCrop(output_size),
                                     AsTorchBatch(2))

        self.output_size = output_size

    def __len__(self):
        return self.data.len_segmentation()

    def __getitem__(self, idx):
        img, gt = self.data.get_segmentation(idx)
        img, gt = img.astype(np.float32), gt.astype(np.float32)
        if self.transform is not None:
            img, gt = self.transform(img, gt)
        return img, gt


class CTCSemisupervisedSegmentationDataset(CTCSegmentationDataset):
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

            img, seg = self.data.get_segmentation(i)

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
        import pdb
        pdb.set_trace()

        del shape_feats

    def __len__(self):
        return self.data.len_images()

    def __getitem__(self, idx):
        img, gt = self.data.get_allimages(idx)
        img, gt = img.astype(np.float32), gt.astype(np.float32)
        if self.transform is not None:
            img, gt = self.transform(img, gt)
        return img, gt, img, self.means, self.stds


if __name__ == '__main__':
    ds = CTCSegmentationDataset("/mnt/data1/swolf/CTC/Fluo-N3DH-SIM")

    print("start")
    for epoch in range(10):
        for i, (img, seg) in enumerate(ds):
            print(i, len(ds))
