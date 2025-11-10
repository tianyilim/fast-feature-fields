"""
Check that the dataloader is working as expected.

Will visualize events from the Train dataloader.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from f3.utils import ev_to_frames
from f3.utils.dataloader import get_dataloaders_from_args

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)-8s: %(message)s')
logger.setLevel(logging.DEBUG)


def main(args):
    logger.info("Starting dataloader...")
    dataset, _ = get_dataloaders_from_args(args, logger, shuffle=True)
    indices_to_choose = np.linspace(
        0, len(dataset), 20, dtype=int, endpoint=False)

    for idx, data in enumerate(tqdm(dataset, dynamic_ncols=True)):
        if idx not in indices_to_choose:
            continue

        ctx, pred, totcnt_ctx, totcnt_pred, valid_mask = data
        events_frame = ev_to_frames(
            ctx, totcnt_ctx,
            args.frame_sizes[0], args.frame_sizes[1]
        )[0].cpu().numpy().T

        plt.figure()
        plt.imshow(events_frame)
        plt.title(f"Frame {idx}/{len(dataset)}")
        plt.axis("off")
        plt.savefig(f"dataset_frame_{idx:06d}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--conf", type=str, required=True,
                        help="Path to the config file. If provided will load configs from here." +
                        "Rest of the configs will be loaded from the defaults below")
    args = parser.parse_args()
    keys = set(vars(args).keys())

    with open(args.conf, "r") as f:
        conf = yaml.safe_load(f)
    for key, value in conf.items():
        if key not in keys:
            setattr(args, key, value)

    main(args)
