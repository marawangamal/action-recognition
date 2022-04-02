""" Extracts frames root directory containing videos.

    Example Usage:
        python videoprocessing/extract_frames.py --root ucf/ucf_vids --out ucf/ucf_imgs \
        --vid_ext .mp4 --imgs_ext .jpg -r 30
"""

import os
import os.path as osp
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from arcore.utils import get_files_list


class FrameExtractor:
    """Extracts frames from video files.

    Args:
        in_rootdir (str): dir containing many `videofile.api` to be processed
        out_rootdir (str): directory to contain frames folders
        video_extension (str): video file extension type (i.e. .avi, .mp4)
        img_extension (str): img file extension type (i.e. .jpg, .png)
        frame_rate (int):

    """

    def __init__(self, in_rootdir, out_rootdir, vid_extension=".avi", img_extension="%05d.jpg", frame_rate=16,
                 overwrite=False):
        self.in_rootdir = in_rootdir
        self.out_rootdir = out_rootdir
        self.vid_extension = vid_extension
        self.img_extension = img_extension
        self.frame_rate = frame_rate
        self.overwrite = overwrite

        assert "%" in img_extension, "Invalid image template given (Example valid tamplate: ``%05d.jpg``)"

    def __call__(self, video_name, in_path_from_root="", out_path_from_root=""):
        in_path = osp.join(self.in_rootdir, in_path_from_root, video_name + self.vid_extension)
        out_path = osp.join(self.out_rootdir, out_path_from_root, video_name, self.img_extension)

        if not osp.exists(osp.join(self.out_rootdir, out_path_from_root, video_name)):
            os.makedirs(osp.join(self.out_rootdir, out_path_from_root, video_name))

        # Dont overwrite
        if not len(os.listdir(osp.join(self.out_rootdir, out_path_from_root, video_name))) > 1 or self.overwrite:
            devnull = open(os.devnull, 'w')
            subprocess.call(['ffmpeg', '-y',
                             '-i', in_path,
                             '-r', str(self.frame_rate),
                             out_path], stdout=devnull, stderr=devnull)


def extract_frames(in_rootdir, out_rootdir, vid_ext, img_ext="%05d.jpg", frame_rate=16, workers=1):
    # Verify ext input
    vid_ext = "." + vid_ext if "." not in vid_ext else vid_ext
    img_ext = "." + img_ext if "." not in img_ext else img_ext

    # Finds all files in subdirectories of root with extension `ext`
    videofolder_paths, videofolder_names = get_files_list(in_rootdir, vid_ext)
    videofolder_names = [v.split(".")[0] for v in videofolder_names]
    videofolder_parents = [v.split("/")[-1] for v in videofolder_paths]

    # Instantiate frame extractor (uses ffmpeg)
    frame_extractor = FrameExtractor(in_rootdir, out_rootdir, vid_ext, img_ext, frame_rate)

    if workers > 1:
        print("Multiprocessing (Num Workers: {})".format(workers))
        pool = Pool(workers)
        # for _ in tqdm(pool.imap_unordered(frame_extractor, videofolder_filepaths), total=len(videofolder_filepaths)):
        #     pass
        for _ in tqdm(pool.starmap(frame_extractor, zip(videofolder_names, videofolder_parents, videofolder_parents)),
                      total=len(videofolder_paths)):
            pass
        pool.close()

    else:
        for i in tqdm(range(len(videofolder_paths))):
            frame_extractor(videofolder_names[i], videofolder_parents[i], videofolder_parents[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/something-something/20bn-something-something-v2")
    parser.add_argument('--out', type=str, default="data/something-something/imgs")
    parser.add_argument('--vid_ext', type=str, default=".mp4")
    parser.add_argument('--img_ext', type=str, default=".jpg")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('-r', type=str, default='16', help="frame rate")
    args = parser.parse_args()

    extract_frames(args.root, args.out, args.vid_ext, args.img_ext, args.workers, args.r)
