import h5py
import numpy as np
from skimage.io import imread
import logging
import vigra
from pathlib import Path
import os
# TEST = False
DATA_SOURCE = 'UNSUPERVISED'
WITH_TRACKING = False
UNSUPERVISED = True


# split_string = "test" if TEST else "train"

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    split_dict = {"unsupervised": ("challenge-datasets", "training-datasets"),
                  "test": ("challenge-datasets",),
                  "train": ("training-datasets",)}

    base_path = "/export/home/swolf/local/data/CTC/data.celltrackingchallenge.net"
    lookup_root_folder = base_path + "/training-datasets/"

    for data_source, subfolder_strings in split_dict.items():
        for dataset in os.listdir(lookup_root_folder):

            print(dataset)

            h5file = f"{base_path}/{data_source}_{dataset}.h5"
            with h5py.File(h5file, 'w') as out:

                for source_str in subfolder_strings:

                    logger.info(f'creating h5 files for {h5file}')

                    root_folder = f"/export/home/swolf/local/data/CTC/data.celltrackingchallenge.net/{source_str}"
                    dataset_folder = Path(root_folder + "/" + dataset)

                    matchstr = '*_GT' if 'training-datasets' == source_str else '*'
                    print(dataset_folder, matchstr)
                    
                    for folder in sorted(dataset_folder.glob(matchstr)):
                        seg_exists = folder.joinpath('SEG').exists()
                        tra_exists = folder.joinpath('TRA').exists()
                        if not tra_exists:
                            logger.info(f'No tracking ground truth found in {folder}')
                        if not seg_exists:
                            logger.info(f'No segmentation ground truth found in {folder}')

                        print(folder)
                        folder_index = folder.stem[:2]
                        img_dir = dataset_folder / folder_index

                        full_img_set = vigra.impex.readVolume(str(img_dir / 't000.tif'))[..., 0].transpose(2, 1, 0)
                        gt_mask = np.zeros((len(full_img_set),), dtype=bool)
                        get_segmentation = []

                        if not WITH_TRACKING:

                            for t in range(len(full_img_set)):
                                seg_file = folder.joinpath(f'SEG/man_seg{t:03d}.tif')
                                if seg_file.exists():
                                    gt_mask[t] = True
                                    seg_gt = imread(seg_file)
                                    get_segmentation.append(seg_gt)
                                else:
                                    gt_mask[t] = True
                                    seg_gt = np.zeros(full_img_set[0].shape, dtype=np.uint32)
                                    get_segmentation.append(seg_gt)

                            print("folder index", folder_index, full_img_set.shape)

                            # convert to usual format
                            img_data = full_img_set[gt_mask][None].astype(np.float32) / 255.
                            full_gt_set = np.stack(get_segmentation, axis=0)
                            seg_data = full_gt_set[None].astype(np.uint32)

                            logger.info(f'Saving {source_str[0]}{folder_index} to {h5file}')
                            chunks = (1, 1) + img_data.shape[-2:]
                            out.create_dataset(f'{source_str[0]}{folder_index}/raw', data=img_data, chunks=chunks, compression='gzip')
                            out.create_dataset(f'{source_str[0]}{folder_index}/gt_tracklets', data=seg_data, chunks=chunks, compression='gzip')

                        else:

                            print(full_img_set.shape)
                            full_gt_set = np.zeros(full_img_set.shape, dtype=np.int32)

                            true_background_id = -1
                            single_object_id = 2

                            division_file = folder.joinpath(f'TRA/man_track.txt')
                            division_transitions = []
                            with open(division_file) as f:
                                lines = f.readlines()
                                for s in lines:
                                    trans = s.split(" ")
                                    division_transitions.append([int(x) for x in trans])

                            out.create_dataset(f'{folder_index}/div',
                                               data=np.asarray(division_transitions),
                                               compression='gzip')

                            for t in range(len(full_img_set)):
                                logger.info(f"processing {t}")
                                matched_all_segments = False

                                seg_file = folder.joinpath(f'SEG/man_seg{t:03d}.tif')
                                tra_file = folder.joinpath(f'TRA/man_track{t:03d}.tif')
                                # img = imread(img_file).astype(float)

                                if tra_file.exists():
                                    full_gt_set[t] = imread(tra_file)

                                if seg_file.exists() and not tra_file.exists():
                                    full_gt_set[t] = -1 * imread(seg_file)
                                elif seg_file.exists():
                                    seg_gt = imread(seg_file)

                                    # first necessary condition for complete match of seg and tra objects:
                                    # the numbers of unique segments must be equal
                                    matched_all_segments = len(np.unique(seg_gt)) == len(np.unique(full_gt_set[t]))

                                    for idx in np.unique(seg_gt):
                                        if idx == 0:
                                            continue

                                        mask = seg_gt == idx
                                        matchids = np.unique(full_gt_set[t][mask]).tolist()
                                        if 0 in matchids:
                                            matchids.remove(0)

                                        if len(matchids) == 0:
                                            # one non-matching tracking object found
                                            # therefore we can not trust that label 0
                                            # is only background
                                            matched_all_segments = False
                                            full_gt_set[t][mask] = -single_object_id
                                            single_object_id += 1
                                        elif len(matchids) == 1:
                                            # a unique tracking object was found
                                            tra_idx = matchids[-1]
                                            full_gt_set[t][mask] = tra_idx
                                        else:
                                            # chose object with highest overlapp
                                            print("warning: need to resolve overlapping tracking and segmentation gt")
                                            print(matchids)
                                            best_id = 0
                                            best_overlap = 0
                                            for mid in matchids:
                                                overlap = (full_gt_set[t][mask] == mid).sum()
                                                if overlap > best_overlap:
                                                    best_overlap = overlap
                                                    best_id = mid
                                            print(f" resolved: {best_id}, count: {best_overlap}")
                                            full_gt_set[t][mask] = best_id

                                if matched_all_segments:
                                    full_gt_set[t][full_gt_set[t] == 0] = true_background_id

                            print("folder index", folder_index, full_img_set.shape)
                            chunks = (1, 1) + full_img_set.shape[-2:]

                            # convert to usual format
                            img_data = full_img_set[None].astype(np.float32) / 255.
                            seg_data = full_gt_set[None].astype(np.uint32)

                            out.create_dataset(f'{folder_index}/img', data=img_data, chunks=chunks, compression='gzip')
                            out.create_dataset(f'{folder_index}/gt_tracklets', data=seg_data, chunks=chunks, compression='gzip')
                            logger.info(f'Saved {folder_index} to {h5file}')
