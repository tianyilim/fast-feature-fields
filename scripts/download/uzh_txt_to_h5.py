"""
This script processes the text from the UZH-FPV dataset and converts it into HDF5 format.
"""

import argparse
from pathlib import Path

import h5py
from tqdm import tqdm


def main(args):
    for txt_parent in args.uzh_save_path.iterdir():
        if not (txt_parent / "events.txt").exists():
            continue

        print()
        print("==" * 40)
        print(f"Processing {txt_parent.name}...")
        h5_out_file = txt_parent / f"{txt_parent.stem}.h5"

        if h5_out_file.exists():
            print(
                f"HDF5 file {h5_out_file} already exists, skipping processing.")
            continue

        print("Will write to", h5_out_file)

        timestamps = []
        xs = []
        ys = []
        pols = []
        file_valid = True

        with open(txt_parent / "events.txt", 'r') as f:
            nlines = sum(1 for _ in f)
        with open(txt_parent / "events.txt", 'r') as f:
            for line in tqdm(f, total=nlines, dynamic_ncols=True):
                line = line.strip()
                if "#" in line:
                    continue

                try:
                    timestamp, x, y, pol = line.split()
                except:
                    print(
                        "Invalid data read in file! Pessimistically not reading this file.")
                    print(line)
                    file_valid = False
                    break

                timestamp_us = int(float(timestamp) * 1e6)
                timestamps.append(timestamp_us)
                xs.append(int(x))
                ys.append(int(y))
                pols.append(int(pol))

        if not file_valid:
            continue

        len_data = len(timestamps)
        assert len_data == len(xs) == len(ys) == len(pols)
        print(f"Read {len_data} events.")
        # timestamps start will start with zero
        timestamps = [t-timestamps[0] for t in timestamps]

        ds_duration = timestamps[-1]-timestamps[0]
        assert ds_duration == timestamps[-1] # sanity check that zero-starting timestamps are handled
        print(f"Read dataset {ds_duration/1e6:.3f}s long.")

        # Write to h5 file.
        with h5py.File(h5_out_file, 'w') as h5f:
            h5f.create_dataset("events/t", data=timestamps)
            h5f.create_dataset("events/x", data=xs)
            h5f.create_dataset("events/y", data=ys)
            h5f.create_dataset("events/p", data=pols)

        print("Event data written to", h5_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert UZH-FPV text data to HDF5 format.")
    parser.add_argument("uzh_save_path", type=Path,
                        help="Path to the UZH-FPV dataset directory.")
    args = parser.parse_args()
    main(args)
