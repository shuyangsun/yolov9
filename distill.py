import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-model-coco", type=str, help="COCO 80 class .pt model path"
    )
    parser.add_argument(
        "--teacher-model-face", type=str, help="WIDER face .pt model path"
    )
    parser.add_argument(
        "--inputs", type=str, nargs="+", help="list of directories of images"
    )
    parser.add_argument(
        "--student-config",
        type=str,
        help="name of student config, must be one of the file names in models/detect",
    )
