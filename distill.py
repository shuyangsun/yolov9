import argparse
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size

TEACHER_IMG_SIZE: int = 640

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-coco",
        type=str,
        required=True,
        help="Path to COCO 80 class .pt model.",
    )
    parser.add_argument(
        "--teacher-face", type=str, required=True, help="Path to face .pt model."
    )
    parser.add_argument(
        "--student-cfg",
        type=str,
        required=True,
        help="Path to student config, must be one of the files in models/detect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size per iteration.",
    )
    parser.add_argument(
        "--inputs", type=str, nargs="+", help="list of directories of images"
    )
    args = parser.parse_args()

    DEVICE: str = "cuda:0"

    t_coco = (
        DetectMultiBackend(
            args.teacher_coco, dnn=False, data="data/coco.yaml", fp16=False
        )
        .to(DEVICE)
        .half()
    )

    t_face = (
        DetectMultiBackend(
            args.teacher_face, dnn=False, data="data/face.yaml", fp16=False
        )
        .to(DEVICE)
        .half()
    )
    t_stride, t_names = t_coco.stride, t_coco.names
    img_size = check_img_size(TEACHER_IMG_SIZE, s=t_stride)  # check image size

    dataset = LoadImages(
        path=args.inputs,
        img_size=img_size,
    )

    t_coco.warmup(imgsz=(args.batch_size, 3, img_size, img_size))  # warmup
    t_face.warmup(imgsz=(args.batch_size, 3, img_size, img_size))  # warmup

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(t_coco.device)
        im = im.half().to(DEVICE)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred_coco = t_coco(im, augment=False, visualize=False)[0]
        pred_face = t_face(im, augment=False, visualize=False)[0][1]
        pred_coco_filtered = torch.concatenate(
            [
                pred_coco[:, 0:4, :],  # bbox
                pred_coco[:, 5:9, :],  # person,bicycle,car,motorcycle
                pred_coco[:, 10:11, :],  # bus
                pred_coco[:, 12:13, :],  # truck
                pred_coco[:, 21:22, :],  # dog
                pred_coco[:, 29:32, :],  # backpack,umbrella,handbag
                pred_coco[:, 33:34, :],  # suitcase
            ],
            dim=1,
        )
        pred = torch.concatenate([pred_face, pred_coco_filtered], dim=1)

        # If any cls confidence is higher than face, replace face bbox with coco bbox.
        should_override = (
            torch.max(pred_coco_filtered[:, 4:, :], dim=1)[0] > pred_face[:, 4, :]
        ).bool()
        should_override = torch.reshape(
            should_override, (args.batch_size, 1, should_override.shape[-1])
        )
        bbox_override_mask = should_override.repeat(1, 4, 1)
        pred[:, :4, :][bbox_override_mask] = pred_coco[:, :4, :][bbox_override_mask]
        print(pred.shape)
        exit()
