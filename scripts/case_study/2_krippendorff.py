import argparse
from util import plot
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    metric_order = [
        "Cov",
        "MaxSens",
        "Del - MoRF - blurring",
        "Del - MoRF - constant",
        "Del - MoRF - random",
        "Del - LeRF - blurring",
        "Del - LeRF - constant",
        "Del - LeRF - random",
        "Ins - MoRF - blurring",
        "Ins - MoRF - constant",
        "Ins - MoRF - random",
        "Ins - LeRF - blurring",
        "Ins - LeRF - constant",
        "Ins - LeRF - random",
        "IROF - MoRF - blurring",
        "IROF - MoRF - constant",
        "IROF - MoRF - random",
        "IROF - LeRF - blurring",
        "IROF - LeRF - constant",
        "IROF - LeRF - random",
        "SensN - blurring",
        "SensN - constant",
        "SensN - random",
        "SegSensN - blurring",
        "SegSensN - constant",
        "SegSensN - random",
        "INFD - SQ",
        "INFD - BL",
        "MSIns - blurring",
        "MSIns - constant",
        "MSIns - random",
        "MSDel - blurring",
        "MSDel - constant",
        "MSDel - random",
    ]
    plot.generate_krippendorff_alpha_bar_plot_single_dataset(
        args.in_dir,
        os.path.join(args.out_dir, "krip_bar.svg"),
        color_thresh=0.3,
        metric_order=metric_order,
        metric_selection="all"
    )