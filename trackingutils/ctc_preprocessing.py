import h5py
import numpy as np

import logging
from pathlib import Path

# DATASET = "DIC-C2DH-HeLa"
DATASET = "Fluo-N2DH-SIM+"
# DATASET = "Fluo-N2DL-HeLa"
# TEST = False
TEST = False

subfolder_string = "challenge-datasets" if TEST else "training-datasets"
ROOT_FOLDER = f"/export/home/swolf/local/src/data/volumes/CTC/data.celltrackingchallenge.net/{subfolder_string}/{DATASET}"
ROOT_FOLDER = Path(ROOT_FOLDER)

split_string = "test" if TEST else "train"
H5FILE = f"data/{split_string}_{DATASET}.h5"

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f'creating h5 files for {ROOT_FOLDER.stem}')
    from skimage.io import imread
    import vigra

    with h5py.File(H5FILE, 'w') as out:
        matchstr = '*' if TEST else '*_GT'
        print(ROOT_FOLDER, matchstr)
        for folder in sorted(ROOT_FOLDER.glob(matchstr)):
            seg_exists = folder.joinpath('SEG').exists()
            tra_exists = folder.joinpath('TRA').exists()
            if not tra_exists:
                logger.info(f'No tracking ground truth found in {folder}')
            if not seg_exists:
                logger.info(f'No segmentation ground truth found in {folder}')

            folder_index = folder.stem[:2]
            img_dir = ROOT_FOLDER / folder_index

            full_img_set = vigra.impex.readVolume(str(img_dir / 't000.tif'))[..., 0].transpose(2, 1, 0)

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

            print("folder index", folder_index)
            out.create_dataset(f'{folder_index}/img', data=full_img_set, compression='gzip')
            out.create_dataset(f'{folder_index}/gt', data=full_gt_set, compression='gzip')
            logger.info(f'Saved {folder_index} to {H5FILE}')
