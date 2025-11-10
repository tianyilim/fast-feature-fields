import h5py
import argparse
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--data_h5", required=True, type=str, help="H5 file path with sensor data or the parent folder in case of dsec")
parser.add_argument("--bucket", type=int, default=20, help="Bucket size in us")
parser.add_argument("--dataset", type=str, default="m3ed", choices=["m3ed", "dsec", "mvsec"], help="Dataset name")

args = parser.parse_args()


def gen_ts(camera: str):
    if args.dataset == "m3ed":
        events_t = h5py.File(args.data_h5, 'r')[f'/prophesee/{camera}/t']
    elif args.dataset == "dsec":
        events_t = h5py.File(f'{args.data_h5}/events/{camera}/events.h5', 'r')['events/t']
    elif args.dataset == "mvsec":
        events_t = h5py.File(args.data_h5, 'r')[f'/davis/{camera}/events/t']
    elif args.dataset == "uzhfpv":
        events_t = h5py.File(args.data_h5, 'r')[f'/events/t']
    else:
        raise ValueError("Invalid dataset")

    assert isinstance(events_t, h5py.Dataset)

    assert events_t[0] == 0, "First timestamp is not zero. This breaks some of the assumptions in this repo."

    timeblocks = int(args.bucket)
    FREQ = 1e6/timeblocks
    sequence_duration_us = events_t[-1]
    num_ts = int(sequence_duration_us/timeblocks)

    print("Frequency generated (in Hz): ", FREQ)
    print(f"Total time (in s): {sequence_duration_us/1e6:.3f}")

    def return_index(start_index, till_when):
        start_event_index = start_index
        found_end = False
        counter = 0
        while not found_end:
            end_event_index = np.searchsorted(events_t[start_event_index+counter*1000:start_event_index+(counter+1)*1000], till_when)
            if end_event_index == 1000:
                counter += 1
            else:
                found_end = True
        end_event_index = start_event_index + counter*1000 + end_event_index
        return end_event_index

    TS = np.zeros((num_ts,), dtype=np.uint64)
    start_index = 0
    for block in tqdm(range(0, num_ts)):
        end_index = return_index(start_index, timeblocks*block)
        start_index = end_index
        TS[block] = end_index
    return TS


def main():
    if args.dataset == "m3ed":
        path = args.data_h5.rsplit("/", 1)[0]
        name = args.data_h5.split("/")[-1].replace(".h5", "")
    elif args.dataset == "dsec":
        path = args.data_h5
        name = args.data_h5.split("/")[-1]
    elif args.dataset == "mvsec":
        path = args.data_h5.rsplit("/", 1)[0]
        name = args.data_h5.split("/")[-1].replace(".hdf5", "")
    elif args.dataset == "uzhfpv":
        path = args.data_h5.rsplit("/", 1)[0]
        name = args.data_h5.split("/")[-1].replace(".h5", "")
    else:
        raise ValueError("Invalid dataset")

    if not args.dataset == "uzhfpv":
        TS_LEFT = gen_ts("left")
        TS_RIGHT = gen_ts("right")
        TS = {"left": TS_LEFT, "right": TS_RIGHT}
    else:
        TS_LEFT = gen_ts("") # This dataset is monocular, but we just use "left" as convention
        TS = {"left": TS_LEFT}

    np.save(f"{path}/50khz_{name}.npy", TS)


if __name__ == "__main__":
    main()
