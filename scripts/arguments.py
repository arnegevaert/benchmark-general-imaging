import argparse


def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples-dataset", type=str, required=True)
    parser.add_argument("-a", "--attributions-dataset", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-o", "--output-file", type=str)
    return parser
