from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.parse_args(
        "--input_dir",
        type=str,
        help="input the path to the dir containing the images"
    )