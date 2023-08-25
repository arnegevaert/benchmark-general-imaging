import h5py
from tqdm import tqdm
import argparse
import os
from attribench.result import MetricResult


def merge_attributions(
    src_file: h5py.File, dst_file: h5py.File, out_file: h5py.File
):
    # First copy everything from the destination file that is not present in the source file
    for method in dst_file.keys():
        method_dataset = dst_file[method]
        assert isinstance(method_dataset, h5py.Dataset)
        if method not in src_file.keys():
            out_file.create_dataset(method, data=method_dataset[:])

    # Then copy everything from the source file
    for method in src_file.keys():
        method_dataset = src_file[method]
        assert isinstance(method_dataset, h5py.Dataset)
        out_file.create_dataset(method, data=method_dataset[:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge results from two directories. Results from the source"
        " directory will be merged into the destination directory."
        " In case of overlap, the source results will be used."
        " The results in the source and destination directories are assumed"
        " to only be different in the methods used."
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        help="Path to the directory containing the source results."
        " In case of overlap, the source results will be used.",
    )
    parser.add_argument(
        "--dst-dir",
        type=str,
        help="Path to the directory containing the destination results."
        " In case of overlap, the destination results will be overwritten by"
        " the source results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Path to the directory where the final results will be saved.",
    )
    args = parser.parse_args()

    datasets = os.listdir(args.src_dir)
    if os.listdir(args.dst_dir) != datasets:
        raise ValueError(
            "Source and destination directories must have the same datasets. "
            f"Found {datasets} in source directory "
            f"and {os.listdir(args.dst_dir)} in destination directory."
        )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    prog = tqdm(datasets)
    for ds in prog:
        prog.set_description(f"Processing dataset {ds}")
        if not os.path.exists(os.path.join(args.out_dir, ds)):
            os.makedirs(os.path.join(args.out_dir, ds))
        else:
            # Remove contents of out dir
            for filename in os.listdir(os.path.join(args.out_dir, ds)):
                os.remove(os.path.join(args.out_dir, ds, filename))

        if os.listdir(os.path.join(args.src_dir, ds)) != os.listdir(
            os.path.join(args.dst_dir, ds)
        ):
            raise ValueError(
                "Source and destination directories must have the same files."
                f"Found {os.listdir(os.path.join(args.src_dir, ds))} "
                "in source directory "
                f"and {os.listdir(os.path.join(args.dst_dir, ds))} "
                "in destination directory."
            )
        # Merge attribution file
        src_attrs_file = h5py.File(
            os.path.join(args.src_dir, ds, "attributions.h5"), "r"
        )
        dst_attrs_file = h5py.File(
            os.path.join(args.dst_dir, ds, "attributions.h5"), "r"
        )
        out_attrs_file = h5py.File(
            os.path.join(args.out_dir, ds, "attributions.h5"), "w"
        )
        merge_attributions(src_attrs_file, dst_attrs_file, out_attrs_file)

        # Merge all result files
        result_files = [
            filename
            for filename in os.listdir(os.path.join(args.src_dir, ds))
            if filename not in ["samples.h5", "attributions.h5"]
        ]
        for filename in result_files:
            src_result = MetricResult.load(
                os.path.join(args.src_dir, ds, filename)
            )
            dst_result = MetricResult.load(
                os.path.join(args.dst_dir, ds, filename)
            )
            dst_result.merge(src_result, level="method", allow_overwrite=True)
            dst_result.save(
                os.path.join(args.out_dir, ds, filename), format="hdf5"
            )
