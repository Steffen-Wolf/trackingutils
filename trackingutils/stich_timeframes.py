from datasets import CTCSegmentationDataset, CTCDataWrapper
import h5py
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    root_folder = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC"
    output_file = "/mnt/data1/swolf/CTC/Fluo-N3DL-TRIC/tracking_approx.h5"
    ndim = 3

    crops = {"02": [slice(None), slice(150, 1500), slice(100, 1500)],
             "01": [slice(None), slice(500, 2100), slice(140, 1150)]}

    # root_folder = "/mnt/data1/swolf/CTC/PhC-C2DL-PSC"
    # ndim = 2

    number_of_frames = {1: 65, 2: 210}

    ds = CTCDataWrapper(root_folder, load_3d=ndim == 3, normalize=False)

    with h5py.File(f"{root_folder}/data.h5", "w") as h5file:

        for index in tqdm(range(ds.len_tracking())):
            folder_name = str(ds.get_tracking(index, to_numpy=False)[0]).split("/")[-2]
            folder_number, t = ds.tracking_files[index][2]
            slice_fn = f"{root_folder}/tracking_approx_{folder_number:02}_{t:03}.h5"

            with h5py.File(slice_fn, "r") as h5slice:
                img = h5slice[f"{folder_number}/img"][:]
                seg = h5slice[f"{folder_number}/tracklet_seg"][:]

            raw_ds = h5file.require_dataset(f"{folder_number:02}/raw",
                                            shape=(number_of_frames[folder_number],) + tuple(img.shape),
                                            dtype=np.float32,
                                            chunks=(1,) + tuple(img.shape))

            seg_ds = h5file.require_dataset(f"{folder_number:02}/tracklet_seg",
                                            shape=(number_of_frames[folder_number],) + tuple(seg.shape),
                                            dtype=np.int32,
                                            chunks=(1,) + tuple(seg.shape))

            raw_ds[t] = img
            seg_ds[t] = seg
