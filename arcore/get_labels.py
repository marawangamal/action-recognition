""" Generates txt file containing labels.

    Example usage:

    python arcore/get_labels.py --root_imgs data/data_ucf/ucf_imgs --root_flow data/data_ucf/ucf_flow --ds ucf --out data/data_ucf
    --test_list_path ucf/ucfTrainTestlist/trainlist02.txt
"""

import argparse
from arcore.datasets import *
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_imgs", default="data/ucf")
    parser.add_argument("--root_flow", default=None)
    parser.add_argument("--ext_imgs", default='.jpg')
    parser.add_argument("--ext_flow", default='.jpg')
    parser.add_argument("--ds", default='ucf')
    parser.add_argument("--out", default="data/ucf", help="output directory")
    parser.add_argument("--test_list_path", default=None, help="test list/split")
    args = parser.parse_args()

    get_labels = {
        'ucf': get_labels_ucf
        # 'something': get_labels_something,
        # 'kinetics': get_labels_kinetics,
    }[args.ds]

    train_records, test_records = get_labels(root_imgs=args.root_imgs, ext_imgs=args.ext_imgs, root_flow=args.root_flow,
                                             ext_flow=args.ext_flow, test_list_path=args.test_list_path)

    # Write to csv
    pd.DataFrame(train_records).to_csv(osp.join(args.out, 'train_labels.txt'), header=False)
    pd.DataFrame(test_records).to_csv(osp.join(args.out, 'train_labels.txt'), header=False)




