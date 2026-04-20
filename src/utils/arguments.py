from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--input_dir",
        type=str,
        help="input the path to the dir containing the images"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="input the path to the output dir"
    )
    
    return parser.parse_args()