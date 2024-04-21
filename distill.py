import argparse
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size

TEACHER_IMG_SIZE: int = 640

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
    args = parser.parse_args()

    t_coco = DetectMultiBackend(
        args.teacher_model_coco, dnn=False, data="data/coco.yaml", fp16=True
    )

    t_face = DetectMultiBackend(
        args.teacher_model_face, dnn=False, data="data/face.yaml", fp16=True
    )
    t_stride, t_names = t_coco.stride, t_coco.names
    img_size = check_img_size(TEACHER_IMG_SIZE, s=t_stride)  # check image size

    dataset = LoadImages(
        path=args.inputs,
        img_size=img_size,
    )

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(t_coco.device)
        im = im.half() if t_coco.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred_coco = t_coco(im, augment=False, visualize=False)[0]
        pred_face = t_face(im, augment=False, visualize=False)[0][1]
        print(pred_coco.shape)
        print(pred_face.shape)
        exit()
