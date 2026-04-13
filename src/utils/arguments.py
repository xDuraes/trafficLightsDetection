from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="input the path to the dir containing the images"
    )
    return parser.parse_args()