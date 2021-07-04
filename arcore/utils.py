import os
import pandas as pd


def get_labels_txt(root, ext, half=False):
    """ Returns dataframe with columns ['path', 'num_frames', 'cls'].

    Args:
        root (str): top level directory that will be searched for files
        ext (str): extension used to match files, E.g., '.jpg'
        half (bool): sometimes when flow is used we have double the number of frames for x and y.

    Returns:
        (pd.DataFrame)
    """

    videofolder_root, videofolder_names = get_files_list(root, ext)
    videofolder_paths = [osp.join(videofolder_root[j], videofolder_names[j]) for j in range(len(videofolder_root))]
    num_frames = []
    for videofolder_path in videofolder_paths:
        imgs = [x for x in os.listdir(videofolder_path) if ext in x]
        num_frames.append(len(imgs) // 2 if half else len(imgs))

    df = pd.DataFrame(zip(videofolder_paths, num_frames))


def get_classes(root, out=None):
    classes_list = os.listdir(root)
    classes_df = pd.DataFrame([[classes_list[i], i] for i in range(len(classes_list))])

    # Write to csv
    if out is not None:
        classes_df.to_csv(out, header=False)

    # Save to a dict
    class2index = {}
    for i, cls in enumerate(classes_list):
        class2index[cls] = i

    return class2index


def get_files_list(top_level_root, ext):
    """ Searches sub-directories of `top_level_root` for files with extension `ext`

    Args:
        top_level_root (str): root to search
        ext (str): extension to look for

    Returns:
        roots_list (list): path to directory containing matched file
        file_names_list (list): name of matched file.

    Note:
        To access a matched file use osp.join(roots_list[i], file_names_list[i])
    """

    roots_list = []
    file_names_list = []
    for root, dirs, files in os.walk(top_level_root):
        for file_name in files:
            if ext in files:
                roots_list.append(root)
                file_names_list.append(file_name)

    return roots_list, file_names_list
