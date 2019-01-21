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

    def __init__(self, root_folder, use_buffer=True):
        # create list of available image files
        self.root_folder = Path(root_folder)
        self.parse_root_folder()
        self._buffer = {}
        self.use_buffer = use_buffer

    def parse_root_folder(self):

        self.image_files = {}
        self.segmentation_files = []
        self.link_files = {}

        for data_folder in sorted(self.root_folder.glob("[0-9][0-9]")):
            folder_number = int(data_folder.name)
            # print(data_folder.name, folder_number)
            self.image_files[folder_number] = {}
            self.link_files[folder_number] = {}

            for image_file in sorted(data_folder.glob("t[0-9][0-9][0-9].tif")):
                t = int(image_file.name[1:4])
                self.image_files[folder_number][t] = {}
                self.image_files[folder_number][t]["raw_file"] = image_file

                # check for GT folder
                gt_folder = self.root_folder.joinpath(f"{folder_number:02}_GT")
                if gt_folder.exists():
                    seg_file = gt_folder.joinpath("SEG", f"man_seg{t:03}.tif")
                    if seg_file.exists():
                        self.image_files[folder_number][t]["has_segmentation"] = True
                        self.image_files[folder_number][t]["segmentation_file"] = seg_file
                        self.segmentation_files.append((image_file, seg_file))

                    tra_file = gt_folder.joinpath("TRA", f"man_track{t:03}.tif")
                    if tra_file.exists():
                        self.image_files[folder_number][t]["tracking_file"] = tra_file

            link_file = gt_folder.joinpath("TRA", f"man_track.txt")
            if link_file.exists():
                self.image_files[folder_number] = link_file

    @property
    def segmentation(self, to_numpy=True, use_buffer=True):
        for i in range(self.len_segmentation()):
            yield self.get_segmentation(i,
                                        to_numpy=to_numpy)

    def len_segmentation(self):
        return len(self.segmentation_files)

    def get_image(self, file_name, buffer_name=None):
        # create new buffer dictionary if it does not exist
        if self.use_buffer:
            if buffer_name not in self._buffer:
                self._buffer[buffer_name] = {}

        inbuffer = file_name in self._buffer[buffer_name]
        load_from_file = (self.use_buffer and not inbuffer) \
            or not self.use_buffer

        if load_from_file:
            img = vigra.impex.readImage(str(file_name)).transpose(2, 1, 0)

            if self.use_buffer:
                self._buffer[buffer_name][file_name] = img
        else:
            img = self._buffer[buffer_name][file_name]

        return img

    def get_segmentation(self, index, to_numpy=True):
        rf, sf = self.segmentation_files[index]
        if not to_numpy:
            return rf, sf
        else:
            return self.get_image(rf, buffer_name="raw"), \
                self.get_image(sf, buffer_name="seg")


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
        else:
            self.transform = Compose(RandomCrop(output_size),
                                     AsTorchBatch(2))

        self.output_size = output_size

    def __len__(self):
        return self.data.len_segmentation()

    def __getitem__(self, idx):
        img, gt = self.data.get_segmentation(idx)
        img, gt = img.astype(np.float32), gt.astype(np.float32)
        img, gt = self.transform(img, gt)
        return img, gt

if __name__ == '__main__':
    ds = CTCSegmentationDataset("/mnt/data1/swolf/CTC/Fluo-N3DH-SIM")

    print("start")
    for epoch in range(10):
        for i, (img, seg) in enumerate(ds):
            print(i, len(ds))
