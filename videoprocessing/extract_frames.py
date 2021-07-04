""" Extracts frames root directory containing videos.

    Example Usage

    python videoprocessing/extract_frames.py --root ucf/ucf_vids --out ucf/ucf_imgs --vid_ext .mp4 --imgs_ext .jpg -r 30
"""

import os
import os.path as osp
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
from arcore.utils import get_files_list


class FrameExtractor:
    def __init__(self, root, out, video_extension, img_extension, frame_rate):
        self.root = root
        self.out = out  # root dir containing videofolders
        self.video_extension = video_extension
        self.img_extension = img_extension
        self.frame_rate = str(frame_rate)

    def __call__(self, video_name):
        in_path = osp.join(self.root, video_name)
        out_path = osp.join(self.out, video_name)

        if not osp.exists(in_path):
            os.makedirs(out_path)

        devnull = open(os.devnull, 'w')
        subprocess.call(['ffmpeg', '-y', '-i', in_path, '-r', self.frame_rate, out_path], stdout=devnull,
                        stderr=devnull)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/something-something/20bn-something-something-v2")
    parser.add_argument('--out', type=str, default="data/something-something/imgs")
    parser.add_argument('--vid_ext', type=str, default=".mp4")
    parser.add_argument('--img_ext', type=str, default=".jpg")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('-r', type=str, default='16', help="frame rate")
    args = parser.parse_args()

    # Finds all files in subdirectories of root with extension `args.ext`
    videofolder_paths, videofolder_names = get_files_list(args.root, args.ext)
    frame_extractor = FrameExtractor(args.root, args.out, args.vid_ext, args.img_ext, args.r)

    pool = Pool(args.workers)
    for _ in tqdm(pool.imap_unordered(frame_extractor, videofolder_names), total=len(videofolder_names)):
        pass
    pool.close()
