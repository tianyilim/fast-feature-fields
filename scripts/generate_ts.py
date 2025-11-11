import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_h5", required=True, type=str, help="H5 file path with sensor data or the parent folder in case of dsec")
parser.add_argument("--bucket", type=int, default=20, help="Bucket size in us")
parser.add_argument("--dataset", type=str, default="m3ed", choices=["m3ed", "dsec", "mvsec", "uzhfpv"], help="Dataset name")
parser.add_argument("--validate", action="store_true", help="Validate generated timestamps")


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

    assert events_t[0] <= 1000, "First timestamp is not zero. This breaks some of the assumptions in this repo."
    assert np.all(events_t[1:] >= events_t[:-1]), "Timestamps file is not uniformly increasing!"

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
        end_event_index = 0
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
        assert end_index >= start_index, f"{block}/{num_ts}, {start_index=}, {end_index=}" 
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

    out_path = Path(f"{path}/50khz_{name}.npy")
    if out_path.exists():
        print(str(out_path), "exists, not generating.")
    else:
        print("Generating", str(out_path), "for", args.dataset)
        if not args.dataset == "uzhfpv":
            TS_LEFT = gen_ts("left")
            TS_RIGHT = gen_ts("right")
            TS = {"left": TS_LEFT, "right": TS_RIGHT}
        else:
            TS_LEFT = gen_ts("") # This dataset is monocular, but we just use "left" as convention
            TS = {"left": TS_LEFT}

        np.save(out_path,
                TS  # type: ignore
                )

    # Validate existing timestamps
    if args.validate:
        import matplotlib.pyplot as plt
        TS = np.load(out_path, allow_pickle=True).item()["left"]
        time_ctx_us = 20_000
        min_numevents_ctx = 1000
        end_time = len(TS) * 20

        valid_timestamps = []
        invalid_timestamps = []
        start_time = time_ctx_us               # in us
        end_time = int(end_time - time_ctx_us)  # in us
        for t0 in tqdm(range(start_time, end_time, time_ctx_us), desc="Generating valid timestamps"):
            # Count: The number of events inside the context window
            # Same logic found in dataloader.py
            cnt = TS[t0 // 20] - TS[((t0 - time_ctx_us) // 20)]
            if cnt > 0 and (cnt-1) >= min_numevents_ctx:
                valid_timestamps.append(t0)
            else:
                invalid_timestamps.append(t0)

        # Number of data points we have for training and testing
        numblocks = len(valid_timestamps)
        print(f"{numblocks} valid blocks found!")

        plt.figure()
        for invalid_t in invalid_timestamps:
            plt.axvline(x=invalid_t, c='tab:orange', alpha=0.25)
        plt.axvline(x=valid_timestamps[0])
        plt.axvline(x=valid_timestamps[-1])

        plt.grid()

        timestamp = []
        end_index = []

        for t0 in tqdm(valid_timestamps, desc="Validating points"):
            assert t0 > time_ctx_us, f"query time {t0=} is smaller than context window!"
            _e_time_index = t0 // 20
            ei = np.int64(TS[_e_time_index] - 1)
            _s_time_index = ((t0 - time_ctx_us) // 20) + 1
            si = np.int64(TS[_s_time_index])

            # If this shows, then we have a problem.
            if not ei >= si:
                print(f"{_s_time_index=}")
                print(f"{_e_time_index=}")
                print(f"{si=}")
                print(f"{ei=}")
                print(len(TS))
                plt.axvline(x=t0, c='r')

            timestamp.append(t0)
            end_index.append(ei)

        plt.plot(timestamp, end_index, '.-', label="Timestamp")
        plt.legend()
        plt.savefig(f"{name}_validation.png")


if __name__ == "__main__":
    main()
