import pandas as pd
import os
import os.path as osp


def get_labels_ucf(root_imgs, ext_imgs, root_flow=None, ext_flow=None, half=False, test_list_path=None):
    """ Returns DataFrame with columns ['imgs_path', 'nframes', 'flow_path', 'nframes', 'cls']

    Args:
        root_imgs (str):
        ext_imgs (str): extension to match against
        root_flow (str):
        ext_flow (str):
        half (bool):
       test_list_path (str):

    Returns:
        train_records (list):
        test_records (list):
    """
    videofolder_roots_imgs, videofolder_names_imgs = get_files_list(root_imgs, ext_imgs)
    videofolder_paths_imgs = [osp.join(videofolder_roots_imgs[j], videofolder_names_imgs[j]) for j in
                              range(len(videofolder_roots_imgs))]

    videofolder_classes = [root.split('/')[-1] for root in videofolder_roots_imgs]  # specific to UCF101 dataset

    num_frames_imgs = []
    for videofolder_path_imgs in videofolder_paths_imgs:
        imgs = [x for x in os.listdir(videofolder_path_imgs) if ext_imgs in x]
        num_frames_imgs.append(len(imgs))

    if root_flow is not None:
        num_frames_flow = []
        videofolder_roots_flow, videofolder_names_imgs = get_files_list(root_flow, ext_flow)
        videofolder_paths_flow = [osp.join(videofolder_roots_imgs[j], videofolder_names_imgs[j]) for j in
                                  range(len(videofolder_roots_imgs))]

        for videofolder_path_flow in videofolder_paths_flow:
            imgs = [x for x in os.listdir(videofolder_path_flow) if ext_flow in x]
            num_frames_flow.append(len(imgs) // 2 if half else len(imgs))

        records = zip(videofolder_paths_imgs, num_frames_imgs, videofolder_paths_flow, num_frames_flow,
                      videofolder_classes)

    else:
        records = zip(videofolder_paths_imgs, num_frames_imgs, videofolder_classes)

    # Train/Test Split

    record_ids = [osp.join(videofolder_classes[i], videofolder_names_imgs[i]) for i in range(len(
        videofolder_names_imgs))]

    # Records in `test_list` will be omitted
    if test_list_path is not None:

        # E.g. record in test_list: ['ApplyEyeMakeup/v_ApplyEyeMakeup_g03_c05.avi']
        test_list = pd.read_csv(test_list_path).values.to_list()

    else:
        test_list = []

    test_records = []
    train_records = []
    for i, rec in enumerate(records):
        if record_ids[i] in test_list:
            test_records.append(rec)
        else:
            train_records.append(rec)

    return train_records, test_records
