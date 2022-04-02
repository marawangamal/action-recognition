import subprocess
import os
import csv
import os.path as osp
import argparse

from arcore.videoprocessing import extract_frames
from arcore.utils import load_obj, save_obj
from arcore.utils import save_to_txt


def prep_hmdb(args):
    """Downloads HMDB51 dataset to ``args.data_root``"""

    # Download data
    if not osp.exists(osp.join(str(args.data_root), "rarfiles", "hmdb51_org.rar")):
        subprocess.call(["wget", "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
                        "-P", osp.join(str(args.data_root), "rarfiles")])

    if not osp.exists(osp.join(str(args.data_root), "rarfiles", "test_train_splits.rar")):
        subprocess.call(["wget", "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
                         "-P", osp.join(str(args.data_root), "rarfiles"), "-o-"])

    # Extract - first level produces ``action_name.rar`` files
    subprocess.call(["unrar", "x", "{}".format(osp.join(args.data_root, "rarfiles", "hmdb51_org.rar")),
                     osp.join(str(args.data_root), "rarfiles"),
                     "-o-", '-idq'])
    rar_files = os.listdir(osp.join(str(args.data_root), "rarfiles"))

    # Put vids in ``hmdb_vids``
    if not osp.exists(osp.join(str(args.data_root), "hmdb_vids")):
        os.makedirs(osp.join(str(args.data_root), "hmdb_vids"))

    # Extract - second level extracts each ``action_name.rar`` file
    for rar_file in rar_files:
        if rar_file != "hmdb51_org.rar" and rar_file != "test_train_splits.rar":
            subprocess.call(["unrar", "x", "{}".format(osp.join(args.data_root, "rarfiles", rar_file)),
                             osp.join(args.data_root, "hmdb_vids"),
                             "-o-",  # dont overwrite
                             '-idq'])
        elif rar_file == "test_train_splits.rar":
            subprocess.call(["unrar", "x", "{}".format(osp.join(args.data_root, "rarfiles", rar_file)),
                             args.data_root, '-o-', '-idq'])

    # Extract frames/jpegs
    print("Extracting frames from videos to {} ...".format(osp.join(args.data_root, "hmdb_imgs")))
    extract_frames(in_rootdir=osp.join(args.data_root, "hmdb_vids"),
                   out_rootdir=osp.join(args.data_root, "hmdb_imgs"),
                   vid_ext=".avi",
                   img_ext="%05d.jpg",
                   frame_rate=16)

    # Generate Class IDs
    print("Generating classes - saving to {} ...".format(osp.join(args.data_root, "classes.pkl")))
    classes_list = os.listdir(osp.join(args.data_root, "hmdb_vids"))
    classes_dict = {classes_list[i]: i for i in range(len(classes_list))}
    save_obj(classes_dict, osp.join(args.data_root, "classes.pkl"))

    # Create ``train.txt`` and ``test.txt`` files from ``testTrainMulti_7030_splits`` directory
    build_file_list_hmdb(args.data_root)


def build_file_list_hmdb(rootdir, splitsdir="testTrainMulti_7030_splits", class_id_path="classes.pkl", split_id=1,
                         val_ratio=0.2):
    """Prepares ``train.txt`` and ``test.txt`` using defualt label files.

    Example:
        --action1_split1.txt
            (Eg. row: path/to/videofile.api [0: None, 1: Train, 2: Test])
        --action2_split1.txt
        ...
        --actionD_split1.txt

        ==>

        train1.txt
            (Eg. row: path/to/videofolder start_frame_num end_frame_num class_id )
        val1.txt
        test1.txt
    """

    train_list, test_list, null_list = list(), list(), list()
    class_id_dict = load_obj(osp.join(rootdir, class_id_path))  # Eg. {"action1_name": id1, "action2_name": id2, ...}

    # Each action has a train-test split file (eg. action1_test_split1.txt)
    for video_split_filename in os.listdir(osp.join(rootdir, splitsdir)):
        if str(split_id) not in video_split_filename:
            continue

        class_name = video_split_filename.split("_test_split{}".format(split_id))[0]

        with open(osp.join(rootdir, splitsdir, video_split_filename), 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                video_filename, train_flag = row[0].split(" ")[:-1]
                video_name = video_filename.split(".")[0]
                target_list = {"1": train_list, "2": test_list, "0": null_list}[train_flag]
                video_filepath = osp.join(rootdir, "hmdb_imgs", class_name, video_name)
                num_frames = len(os.listdir(video_filepath))
                class_id = class_id_dict[class_name]

                # Each row contains ``path\to\videofolder start_frame_num end_frame_num class_id``
                if num_frames > 1:
                    target_list.append([video_filepath, str(1), str(num_frames), str(class_id)])
                else:
                    pass

    # Save lists
    val_list = test_list[:int(val_ratio*len(test_list))]
    test_list = test_list[int(val_ratio*len(test_list)):]

    save_to_txt(train_list, osp.join(rootdir, "train{}.txt".format(split_id)))
    save_to_txt(val_list, osp.join(rootdir, "val{}.txt".format(split_id)))
    save_to_txt(test_list, osp.join(rootdir, "test{}.txt".format(split_id)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Action recognition Dataset Downloader')

    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="hmdb")
    args = parser.parse_args()

    # Get data
    {"hmdb": prep_hmdb}[args.dataset](args)

