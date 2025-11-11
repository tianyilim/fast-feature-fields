import h5py
import json
import yaml
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .utils_gen import collate_fn_general, H5WithLazyDivision

from numpy.random import default_rng
rng = default_rng(42)


class BaseExtractor(Dataset):
    """
        Takes in the hdf5 file and timestamps and has functions to get events
        at any time interval
    """
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str,
                 w: int=1280, h: int=720,
                 time_ctx: int=20000, time_pred: int=20000, bucket: int=1000, max_numevents_ctx: int=800000,
                 randomize_ctx: bool=True, camera: str="left",
                 dtype: str="m3ed",
                 randomize_roi: bool=True):
        """
            Args:
                hdf5_file: str
                    The path to the hdf5 file containing the events
                timestamps_50khz_file: str
                    The path to the numpy file containing the timestamps of the events
                    at a 50kHz rate
                w: int
                    The target width, where to center the events
                h: int;
                    The target height, where to center the events
                time_ctx: int
                    The time length of the context. Specified in us
                time_pred: int
                    The time length of the prediction. Specified in us
                bucket: int
                    The time length of each frame. Default is kHz frames. Specified in us
                max_numevents_ctx: int
                    The maximum number of events to sample for the feature field. That is subsample the ctx events
                    if required to this number
                randomize_ctx: bool
                    If True, randomize the context events. If False, return the events in order linspace
                camera: str
                    The camera to use. Can be "left" or "right"
                dtype: str
                    Dataset type. Can be "m3ed" or "dsec" or "mvsec" or "uzhfpv".
                    Changes some of the resolution and path options based on the dataset
                randomize_roi: bool
                    If True, and dataset resolution != target resolution, will fill the ROI randomly.
                    If dataset res < target, will put the dataset somewhere in the frame (not just in center)
                    If dataset res > target, will take a randomized crop of target resolution.
        """
        super(BaseExtractor, self).__init__()
        self.logger = logging.getLogger("__main__")
        self.hdf5_fp = hdf5_file
        self.hdf5_file = h5py.File(hdf5_file, "r")

        ALLOWED_DATASETS = ["m3ed", "dsec", "mvsec", "tartanair-v2"]
        assert dtype in ALLOWED_DATASETS, \
            f"Invalid dataset type {dtype}! Allowed types are {ALLOWED_DATASETS}"

        # Load the timestamps file
        if timestamps_50khz_file is not None:
            self.us_to_discretize = 20  # 50KHz -> 20us per index
            self.timestamps = np.load(timestamps_50khz_file, allow_pickle=True)[()][camera]
            #* Time stamps 50KHz always has timestamps of "where to insert the event" so that the queried time
            #* is >= all the previous timestamps. So the event with the requested time stamp or lesser than that
            #* will be -1 the returned index. Not a bug or a feature, just a note to remember
            self.logger.info(f"Timestamps 50KHz file {timestamps_50khz_file} loaded successfully!")
        else:
            self.us_to_discretize = 1000  # 1KHz -> 1000us per index
            if dtype == "tartanair-v2":
                self.timestamps = self.hdf5_file["ms_to_idx"]
            elif dtype == "m3ed":
                self.timestamps = self.hdf5_file[f"prophesee/{camera}/ms_map_idx"]
            elif dtype == "dsec":
                self.timestamps = self.hdf5_file["ms_to_idx"]
            elif dtype == "mvsec":
                raise ValueError("⚠️ MVSEC h5 files do not have millisecond to event index mapping. So generate the timestamps_50khz_file and use it!")
            self.logger.info(f"Timestamps loaded from hdf5 file for {dtype} dataset successfully -- 'ms_to_idx' is used!")

        # Check if each elem is greater than the elem to its left
        assert np.all(self.timestamps[1:] >= self.timestamps[:-1]), "Timestamps file is not uniformly increasing!"

        if dtype == "m3ed":
            # We use the "camera" event camera
            self.w, self.h = 1280, 720  #! Important: resolution in the dataset
            self.events_x: h5py.Dataset = self.hdf5_file[f"prophesee/{camera}/x"] # type: ignore
            self.events_y: h5py.Dataset = self.hdf5_file[f"prophesee/{camera}/y"] # type: ignore
            self.events_t: h5py.Dataset = self.hdf5_file[f"prophesee/{camera}/t"] # type: ignore
            self.events_p: h5py.Dataset = self.hdf5_file[f"prophesee/{camera}/p"] # type: ignore
        elif dtype == "dsec":
            self.w, self.h = 640, 480   #! Important: resolution in the dataset
            self.events_x: h5py.Dataset = self.hdf5_file[f"events/x"] # type: ignore
            self.events_y: h5py.Dataset = self.hdf5_file[f"events/y"] # type: ignore
            self.events_t: h5py.Dataset = self.hdf5_file[f"events/t"] # type: ignore
            self.events_p: h5py.Dataset = self.hdf5_file[f"events/p"] # type: ignore
        elif dtype == "mvsec":
            self.w, self.h = 346, 260   #! Important: resolution in the dataset
            self.events_x: h5py.Dataset = self.hdf5_file[f"davis/{camera}/events/x"] # type: ignore
            self.events_y: h5py.Dataset = self.hdf5_file[f"davis/{camera}/events/y"] # type: ignore
            self.events_t: h5py.Dataset = self.hdf5_file[f"davis/{camera}/events/t"] # type: ignore
            self.events_p: h5py.Dataset = self.hdf5_file[f"davis/{camera}/events/p"] # type: ignore
        elif dtype == "uzhfpv":
            self.w, self.h = 346, 260   #! Important: resolution in the dataset
            self.events_x: h5py.Dataset = self.hdf5_file[f"events/x"] # type: ignore
            self.events_y: h5py.Dataset = self.hdf5_file[f"events/y"] # type: ignore
            self.events_t: h5py.Dataset = self.hdf5_file[f"events/t"] # type: ignore
            self.events_p: h5py.Dataset = self.hdf5_file[f"events/p"] # type: ignore
        elif dtype == "tartanair-v2":
            self.w, self.h = 640, 640   #! Important: resolution in the dataset
            self.events_x: h5py.Dataset = self.hdf5_file["events/x"] # type: ignore
            self.events_y: h5py.Dataset = self.hdf5_file["events/y"] # type: ignore
            self.events_t: h5py.Dataset = H5WithLazyDivision(self.hdf5_file["events/t"], 1000) # convert ns to us
            self.events_p: h5py.Dataset = self.hdf5_file["events/p"] # type: ignore
        else:
            raise ValueError(f"Invalid dtype: {dtype}! Should be either m3ed or dsec or mvsec or uzhfpv!")

        assert isinstance(self.events_x, h5py.Dataset)
        assert isinstance(self.events_y, h5py.Dataset)
        assert isinstance(self.events_t, h5py.Dataset)
        assert isinstance(self.events_p, h5py.Dataset)

        assert self.events_t[0] < time_ctx, f"First timestamp {self.events_t[0]}>={time_ctx}. This breaks some of the assumptions in this repo."

        self.dtype = dtype
        self.camera = camera
        self.bucket = bucket
        self.time_ctx = time_ctx
        self.time_pred = time_pred
        self.randomize_ctx = randomize_ctx
        self.max_numevents_ctx = max_numevents_ctx
        self.randomize_roi = randomize_roi
        self.metadata = {}

        self.trgt_res = (w, h) #! Important: resolution of the output frame
        self.crop_target = None

        # Boolean mask for the pixels where loss is valid
        self.valid_mask = torch.zeros(self.trgt_res, dtype=torch.bool)

        # Normalization factor for the context events
        self.norm_factor = torch.tensor([
            self.trgt_res[0], self.trgt_res[1], self.time_ctx // self.bucket, 1
        ], dtype=torch.float32)[None, :]

        if self.trgt_res[0] >= self.w and self.trgt_res[1] >= self.h:
            self.logger.info(f"Target res {self.trgt_res}, raw resolution {(self.w, self.h)}, downsampling resolution mode!")
            self.to_downsample = False
        elif self.trgt_res[0] < self.w and self.trgt_res[1] < self.h:
            self.logger.info(f"Target res {self.trgt_res}, raw resolution {(self.w, self.h)}, upsampling resolution mode!")
            self.to_downsample = True
        else:
            raise ValueError(f"Target res {self.trgt_res} must be BOTH larger or smaller than raw resolution {(self.w, self.h)}")


    def save_metadata(self, fname: str="metadata.json"):
        folder_path = Path(self.hdf5_fp).parent
        # if the file doesnt exist, create it
        if not (folder_path / fname).exists():
            with open(folder_path / fname, "w") as f:
                json.dump([self.metadata], f)
                return
        with open(folder_path / fname, "r") as f:
            if f.read() == "":
                json.dump([self.metadata], f)
                return
            else:
                f.seek(0)
                oldmetadata = json.load(f)
        with open(folder_path / fname, "w") as f:
            oldmetadata.append(self.metadata)
            json.dump(oldmetadata, f)

    def get_ctx_fixedtime(self, t0):
        ei = int(self.timestamps[t0 // self.us_to_discretize] - 1)
        si = self.timestamps[(t0 - self.time_ctx) // self.us_to_discretize + 1]

        ctx, self.valid_mask = self._filter_events(si, ei, t0, for_ctx=True)
        totcnt = ctx.shape[0]

        # Downsample total number of input events
        _used = min(totcnt, self.max_numevents_ctx)
        if totcnt > self.max_numevents_ctx:
            if self.randomize_ctx:
                indices = np.sort(rng.choice(totcnt, size=self.max_numevents_ctx, replace=False, shuffle=False))
            else:
                indices = np.linspace(0, totcnt, self.max_numevents_ctx, dtype=np.uint64, endpoint=False)
            ctx = ctx[indices, :]

        _used = ctx.shape[0] # need to update this number after cropping
        #! Ideally we should get rid of the duplicates caused by the bucketing, but it is expensive
        #! And since we work with 1KHz frames, there aren't many duplicates
        ctx = torch.tensor(ctx, dtype=torch.float32) / self.norm_factor
        return ctx, _used

    def get_pred_fixedtime(self, t0):
        si = self.timestamps[t0 // self.us_to_discretize]
        ei = int(self.timestamps[(t0 + self.time_pred) // self.us_to_discretize] - 1)
        pred, _ = self._filter_events(si, ei, t0, for_ctx=False)
        totcnt = pred.shape[0]  # need to update this number after cropping

        if self.dtype == "mvsec": pred[:,3][pred[:,3] == -1] = 0
        pred = torch.tensor(pred, dtype=torch.int32)
        return pred, totcnt

    def _filter_events(self, start_idx: int, end_idx: int, t0, for_ctx: bool):
        '''
        start/end idx: The indices of the event h5 Dataset.
        t0: Start time of the event sequences.
        for_ctx: Whether the events are to be generated for the context or prediction vectors.
              If for context vector, we will use the start time to compute the context window, and generate a center crop.
              Else, we will use the end time to compute the prediction window, and consume the center crop generated by
              the previous filter_events call.

        This means that the get_ctx_fixedtime and get_pred_fixedtime methods MUST be called in an interleaved manner.
        '''

        # Read the raw events
        valid_mask = torch.zeros(self.trgt_res, dtype=torch.bool)
        ret = np.empty((end_idx-start_idx, 4), dtype=np.int32)
        ret[:, 0] = self.events_x[start_idx:end_idx]
        ret[:, 1] = self.events_y[start_idx:end_idx]
        if for_ctx:
            ret[:, 2] = (t0 - self.events_t[start_idx:end_idx]) // self.bucket
        else:
            ret[:, 2] = (self.events_t[start_idx:end_idx] - t0) // self.bucket
        ret[:, 3] = self.events_p[start_idx:end_idx]

        if self.randomize_roi:
            # Randomly sample the top-left corner of the ROI box
            # If upsampling, this will be negative; which amounts to a shift INSIDE the roi
            # If downsampling, this will be positive; which is the expected behaviour
            diff_x = self.w - self.trgt_res[0]
            diff_y = self.h - self.trgt_res[1]
            if for_ctx:
                xs = rng.integers(min(0, diff_x), max(0, diff_x)+1)
                ys = rng.integers(min(0, diff_y), max(0, diff_y)+1)
                self.crop_target = (xs, ys)
            else:
                assert self.crop_target is not None
                xs, ys = self.crop_target
                self.crop_target = None
        else:
            # Take a center crop
            xs = (self.w - self.trgt_res[0]) // 2
            ys = (self.h - self.trgt_res[1]) // 2

        # Crop to valid ROI
        if self.to_downsample:
            ret = self._crop_events(
                ret, xs, ys, self.trgt_res[0], self.trgt_res[1])
            valid_mask[...] = True
        else:
            # Here xs/ys is negative.
            valid_mask[(0-xs):(self.w-xs), (0-ys):(self.h-ys)] = True

        # Shift pixels to target resolution
        ret[:, 0] -= xs
        ret[:, 1] -= ys
        # Check that everything is within the target resolution
        assert np.all(ret[:, 0] >= 0) and np.all(
            ret[:, 0] < self.trgt_res[0]), "X coordinates are out of bounds"
        assert np.all(ret[:, 1] >= 0) and np.all(
            ret[:, 1] < self.trgt_res[1]), "Y coordinates are out of bounds"

        # Valid box post-processing
        if self.dtype == "dsec":
            raw_valid_mask = torch.ones((self.h, self.w), dtype=torch.bool)
            raw_valid_mask[:, self.h-40:self.h] = False  # invalidate bottom 40 rows
            tgt_valid_mask = torch.zeros(self.trgt_res, dtype=torch.bool)

            if self.to_downsample:
                tgt_valid_mask = raw_valid_mask[xs:xs+self.w, ys:ys+self.h]
            else:
                tgt_valid_mask[(0-xs):(self.w-xs), (0-ys):(self.h-ys)] = raw_valid_mask

            valid_mask = torch.bitwise_and(tgt_valid_mask, valid_mask)

        # Postprocessing
        if self.dtype == "mvsec":
            ret[:, 3][ret[:, 3] == -1] = 0

        return ret, valid_mask

    def _crop_events(self, events, x0, y0, w, h):
        valid_x = np.logical_and(events[:, 0] >= x0, events[:, 0] < x0 + w)
        valid_y = np.logical_and(events[:, 1] >= y0, events[:, 1] < y0 + h)
        valid = valid_x & valid_y
        assert valid.shape[0] == events.shape[0], "Valid mask length should be equal to number of events!"

        events = events[valid, :]
        return events


class EventDatasetSingleHDF5(BaseExtractor):
    """
        Loads the dataset and timestamps file and returns the context and prediction events
        + creates metadata for the dataset, ensuring that the context is not too small.
    """
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str,
                 w: int=1280, h: int=720, min_numevents_ctx: int=200000, max_numevents_ctx: int=800000,
                 time_ctx: int=20000, time_pred: int=20000, bucket: int=1000, randomize_ctx: bool=True,
                 camera: str="left", dtype: str="m3ed"):
        """
            Args:
                hdf5_file: str
                    The path to the hdf5 file containing the events
                timestamps_50khz_file: str
                    The path to the numpy file containing the timestamps of the events
                    at a 50kHz rate
                min_numevents_ctx: int
                    The minimum number of events to exist for the feature field training
                max_numevents_ctx: int
                    The maximum number of events to sample for the feature field
                time_ctx: int
                    The time length of the context. Specified in us (events subsampled from this time)
                time_pred: int
                    The time length of the prediction. Specified in us
                bucket: int
                    The time length of each frame. Default is kHz frames. Specified in us
                w, h: int
                    The desired width and height of the frame to rescale to. That is centers the smaller frame in this resolution
                    That is if dataset gives us 640x480 data and w,h is 1280x720, then the 640x480 data is centered in 1280x720
                randomize_ctx: bool
                    If True, randomize the context events. If False, return the events in order linspace
                camera: str
                    The camera to use. Can be "left" or "right"
                dtype: str
                    Dataset type. Can be "m3ed" or "dsec" or "mvsec". Changes some of the resolution and path options based on the dataset
        """
        super(EventDatasetSingleHDF5, self).__init__(hdf5_file, timestamps_50khz_file, w, h, time_ctx, time_pred,
                                                     bucket, max_numevents_ctx, randomize_ctx, camera, dtype)
        self.logger = logging.getLogger("__main__")
        self.min_numevents_ctx = int(min_numevents_ctx * (self.w / w) * (self.h / h)) # rescale the min events based on the camera resolution

        self.process_metadata()
        self.logstats()

    def __len__(self):
        return self.numblocks

    def logstats(self):
        self.logger.info("DATALOADER SPECS:")
        self.logger.info("-"*50)
        self.logger.info(f"Fixed time context mode: {self.time_ctx} us, subsampling max {self.max_numevents_ctx} events")
        self.logger.info(f"Fixed time prediction mode: {self.time_pred} us")
        self.logger.info(f"Timectx: {self.time_ctx}, Timepred: {self.time_pred}, Bucket: {self.bucket} us, Camera: {self.camera}")
        self.logger.info(f"Data Width: {self.w}, Data Height: {self.h}, Randomize ctx: {self.randomize_ctx}")
        self.logger.info(f"Target Width: {self.trgt_res[0]}, Target Height: {self.trgt_res[1]}")
        self.logger.info(f"Number of valid blocks: {self.numblocks}")
        self.logger.info("-"*50 + "\n")

    def process_metadata(self):
        if self.load_metadata():
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            # valid starting points for the feature field to the left and prediction events to the right
            self.valid_0_points = []
            start_time = self.time_ctx                     # in us
            end_time = self.events_t[-1] - self.time_pred  # in us
            for t0 in tqdm(range(start_time, end_time, self.time_ctx), desc="Metadata Pretraining"):
                cnt = self.timestamps[t0 // self.us_to_discretize] - \
                      self.timestamps[(t0 - self.time_ctx) // self.us_to_discretize]
                if cnt > 0 and (cnt-1) >= self.min_numevents_ctx:
                    self.valid_0_points.append(t0)
                    self.logger.info(f"Valid point: {t0}!")
            self.numblocks = len(self.valid_0_points) # number of data points we have for training and testing
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} loaded successfully!: {self.numblocks} valid blocks found!")
            self.metadata = {
                "camera": self.camera,
                "min_numevents_ctx": self.min_numevents_ctx,
                "time_ctx": self.time_ctx,
                "time_pred": self.time_pred,
                #* self.bucket should be here, but we dont remove duplicates, so it doesn't affect validity
                "valid_0_points": self.valid_0_points
            }
            self.save_metadata()

    def load_metadata(self, keys: Optional[list[str]]=None, fname: str="metadata.json"):
        if keys is None:
            keys = ["camera", "min_numevents_ctx", "time_ctx", "time_pred"]
        try:
            with open(Path(self.hdf5_fp).parent / fname, "r") as f:
                for meta in json.load(f):
                    if all(meta.get(key) == getattr(self, key) for key in keys):
                        self.metadata = meta
                        self.valid_0_points = meta["valid_0_points"]
                        self.numblocks = len(self.valid_0_points)
                        return True
        except FileNotFoundError:
            pass
        return False

    def __getitem__(self, idx):
        t0 = self.valid_0_points[idx]
        ctx, totcnt_ctx = self.get_ctx_fixedtime(t0)
        pred, totcnt_pred = self.get_pred_fixedtime(t0)
        return ctx, pred, totcnt_ctx, totcnt_pred, self.valid_mask


#! The following code is pretty redundant for the main feature field and the sub tasks.
def get_dataset_from_h5files(hdf5_files: list[str], timestamps_files: list[str],
                             cameras: Optional[list[str]]=None,
                             dtypes: Optional[list[str]]=None,
                             ranges: Optional[list[tuple[float, int, float]]]=None,
                             **kwargs):
    assert len(hdf5_files) == len(timestamps_files), "The number of hdf5 files and timestamps files should be the same!"
    if cameras is None:
        cameras = ["left"] * len(hdf5_files)   # default to left camera
    if ranges is None:
        ranges = [(0., 1, 1.)] * len(hdf5_files) # default to the whole dataset start, step, stop -> start and stop are fractions
    if dtypes is None:
        dtypes = ["m3ed"] * len(hdf5_files)    # default to m3ed dataset

    datasets = [
        EventDatasetSingleHDF5(hdf5_file, timestamps_file, camera=camera, dtype=dtype, **kwargs)
        for hdf5_file, timestamps_file, camera, dtype in zip(hdf5_files, timestamps_files, cameras, dtypes)
    ]
    subsets = [
        torch.utils.data.Subset(dataset, range(int(len(dataset) * range_[0]), int(len(dataset) * range_[2]), range_[1]))
        for dataset, range_ in zip(datasets, ranges)
    ]
    return torch.utils.data.ConcatDataset(subsets)


def get_dataloader_from_args(args, logger: logging.Logger, shuffle: bool=True, train: bool=True):
    mode = "Training" if train else "Validation"
    config = args.train if train else args.val
    randomize_ctx = args.randomize_ctx if train else False
    logger.info(f"Loading {mode} dataset configs from {config['datasets']}...")

    datasets = []
    for dataset in config["datasets"]:
        with open(dataset, "r") as f:
            datasets.extend(yaml.safe_load(f)['datasets'])
    hdf5_files = [dataset['dataset_path'] for dataset in datasets]
    timestamps_files = [dataset.get('timestamps_path', None) for dataset in datasets]
    cameras = [dataset['camera'] for dataset in datasets]
    dtypes = [dataset['dtype'] for dataset in datasets]
    ranges = [dataset['range'] for dataset in datasets]

    kwargs = {
        "hdf5_files": hdf5_files, "timestamps_files": timestamps_files,
        "min_numevents_ctx": args.min_numevents_ctx, "max_numevents_ctx": args.max_numevents_ctx,
        "time_ctx": args.time_ctx, "time_pred": args.time_pred, "bucket": args.bucket,
        "w": args.frame_sizes[0], "h": args.frame_sizes[1], "randomize_ctx": randomize_ctx,
        "cameras": cameras, "ranges": ranges, "dtypes": dtypes
    }
    logger.info(f"Creating {mode} Dataloaders...")
    dataset = get_dataset_from_h5files(**kwargs)

    persistent_workers = True if train else False
    loader = DataLoader(
        dataset, batch_size=config["mini_batch"], pin_memory=True, persistent_workers=persistent_workers,
        shuffle=shuffle, num_workers=config["num_workers"], prefetch_factor=2*config["batch"]//config["mini_batch"],
        collate_fn=lambda batch: collate_fn_general(batch, [0, 1], [2, 3, 4])
    )
    logger.info(f"{mode} Dataloader created with {len(dataset)} samples!")
    return loader


def get_dataloaders_from_args(args: dict, logger: logging.Logger, shuffle: bool=True):
    train_loader = get_dataloader_from_args(args, logger, shuffle, train=True)
    val_loader = get_dataloader_from_args(args, logger, False, train=False)
    return train_loader, val_loader
