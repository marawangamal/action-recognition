""" Performs optical flow extraction.

    Example usage

    python videoprocessing/extract_flow.py --root data/ucf/ucf101_imgs --out data/ucf/ucf101_flow --method tvl1 --cuda
    --workers 1
"""

from arcore.utils import get_files_list
import os
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import os.path as osp
import numpy as np


class FlowExtractor:
    """
    Optical Flow Extractor.

    Args:
        method (str):
        mean_subtraction (bool): remove mean during discretization
    """

    def __init__(self, method, cuda, flow_tmpl='flow_{}_{:06d}.jpg', mean_subtraction=False):
        self.cuda = cuda
        self.method = method
        self.flow_tmpl = flow_tmpl
        self.mean_subtraction = mean_subtraction
        self.flow_algo = {'tvl1': self.extract_flow_tvl1, 'farneback': self.extract_flow_farneback,
                          'denseflow': self.extract_flow_denseflow}[self.method]

    def extract_flow_tvl1(self, inp):
        """
        Args:
            inp (list): [videofolder_path_imgs, videofolder_path_flow, videofolder_name].
                A videofolder is a folder containing frames of a single video
        """

        videofolder_path_imgs, videofolder_path_flow, videofolder_name = inp
        imgs = sorted(os.listdir(osp.join(videofolder_path_imgs, videofolder_name)))

        for i in range(1, len(imgs)):
            gray1 = cv2.cvtColor(cv2.imread(osp.join(videofolder_path_imgs, videofolder_name, imgs[i - 1])),
                                 cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cv2.imread(osp.join(videofolder_path_imgs, videofolder_name, imgs[i])),
                                 cv2.COLOR_BGR2GRAY)

            if self.cuda:
                gray1_cuda, gray2_cuda = cv2.cuda_GpuMat(gray1), cv2.cuda_GpuMat(gray2)
                flow_tvl1 = optical_flow_cuda.calc(gray1_cuda, gray2_cuda, None).download()  # [W, H, 2]
            else:
                optical_flow = cv2.DualTVL1OpticalFlow_create()
                flow_tvl1 = optical_flow.calc(gray1, gray2, None)

            flow_tvl1_disc = self.discretize(flow_tvl1)

            if not osp.exists(osp.join(videofolder_path_flow, videofolder_name)):
                os.makedirs(osp.join(videofolder_path_flow, videofolder_name))
            cv2.imwrite(osp.join(videofolder_path_flow, videofolder_name, self.flow_tmpl.format('x', i)),
                        flow_tvl1_disc[:, :, 0])
            cv2.imwrite(osp.join(videofolder_path_flow, videofolder_name, self.flow_tmpl.format('y', i)),
                        flow_tvl1_disc[:, :, 1])

    def extract_flow_farneback(self, inp):

        videofolder_path_imgs, videofolder_path_flow, videofolder_name = inp
        imgs = sorted(os.listdir(osp.join(videofolder_path_imgs, videofolder_name)))

        for i in range(1, len(imgs)):
            gray1 = cv2.cvtColor(cv2.imread(os.path.join(videofolder_path_imgs, videofolder_name, imgs[i - 1])),
                                 cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cv2.imread(os.path.join(videofolder_path_imgs, videofolder_name, imgs[i])),
                                 cv2.COLOR_BGR2GRAY)

            # Calculates dense optical flow by Farneback method
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2,
                                                None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            flow_disc = self.discretize(flow)

            if not osp.exists(osp.join(videofolder_path_flow, videofolder_name)):
                os.makedirs(osp.join(videofolder_path_flow, videofolder_name))
            cv2.imwrite(osp.join(videofolder_path_flow, videofolder_name, self.flow_tmpl.format('x', i)),
                        flow_disc[:, :, 0])
            cv2.imwrite(osp.join(videofolder_path_flow, videofolder_name, self.flow_tmpl.format('y', i)),
                        flow_disc[:, :, 1])

    @staticmethod
    def extract_flow_denseflow(inp):
        """ Requires denseflow to be installed """
        videofolder_path_imgs, videofolder_path_flow, videofolder_name = inp
        if not os.path.exists(videofolder_path_flow):
            os.makedirs(videofolder_path_flow)

        # Skip if flow already extracted
        if not os.path.exists(os.path.join(videofolder_path_flow, videofolder_name)) and len(
                os.listdir(os.path.join(videofolder_path_flow, videofolder_name))) > 1:
            devnull = open(os.devnull, 'w')
            subprocess.call(['denseflow', os.path.join(videofolder_path_imgs, videofolder_name), '-b=20', '-a=tvl1',
                             '-s=1', '-if'], stdout=devnull, stderr=devnull)

            # Cleanup - move to correct location
            subprocess.call(['mv', videofolder_name, os.path.join(videofolder_path_flow, videofolder_name)])

    def discretize(self, flow):
        """ Converts `flow` to be in range {0:255}.

        Args:
            flow (np.array float32): [W, H, 2]
        Returns:
            flow_disc (np.array uint8): [W, H, 2]
        """

        W, H, _ = flow.shape
        flow_disc = flow - np.amin(flow.reshape(W * H, 2), axis=0).reshape(1, 1, 2)  # [W, H, 2]
        flow_disc = flow_disc / np.amax(flow_disc.reshape(W * H, 2), axis=0).reshape(1, 1, 2)

        if self.mean_subtraction:
            flow_disc = flow_disc - np.mean(flow_disc.reshape(W * H, 2), axis=0).reshape(1, 1, 2)

        flow_disc = np.rint(flow_disc * 255).astype(int)

        return flow_disc

    def __call__(self, inp):
        self.flow_algo(inp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/ucf/ucf101_imgs", help="directory of videofolders")
    parser.add_argument("--out", type=str, default="data/ucf/ucf101_flow", help="output directory")
    parser.add_argument("--cuda", type=bool, action='store_true', default=False)
    parser.add_argument("--method", type=str, default="tvl1_cuda", choices=['tvl1_cuda', 'denseflow'])
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument('--mean_subtraction', default=False, action='store_true', help='Bool type')
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Build list of input videofolder paths and output videofolder paths
    videofolder_paths_imgs, videofolder_names = get_files_list(args.root, args.ext)
    videofolder_paths_flow = [x.replace(args.root, args.out, 1) for x in videofolder_paths_imgs]

    flow_extractor = FlowExtractor(args.method, args.cuda)

    # Parallel processing.
    pool = Pool(args.workers)
    inputs = [[videofolder_paths_imgs[j], videofolder_paths_flow[j], videofolder_names[j]] for j in
              range(len(videofolder_names))]
    for _ in tqdm(pool.imap_unordered(flow_extractor, inputs), total=len(videofolder_names)):
        pass
    pool.close()
