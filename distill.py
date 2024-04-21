import argparse
import torch

from models.common import DetectMultiBackend
from models.yolo import Model
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

        pred_coco_filtered = torch.cat(
            [
                pred_coco[:, 0:9, :],  # bbox,person,bicycle,car,motorcycle
                pred_coco[:, 9:10, :],  # bus
                pred_coco[:, 11:12, :],  # truck
                pred_coco[:, 20:21, :],  # dog
                pred_coco[:, 28:31, :],  # backpack,umbrella,handbag
                pred_coco[:, 32:33, :],  # suitcase
            ],
            dim=1,
        )

        pred = torch.concatenate([pred_face, pred_coco_filtered[:, 4:, :]], dim=1)

        # If any cls confidence is higher than face, replace face bbox with coco bbox.
        # Remove logits for other classes, as they are incorrect distillation knowledge now.
        should_override_bbox = (
            torch.max(pred_coco_filtered[:, 4:, :], dim=1)[0] > pred_face[:, 4, :]
        ).bool()
        should_override_bbox = torch.reshape(
            should_override_bbox, (args.batch_size, 1, should_override_bbox.shape[-1])
        )
        bbox_override_mask = should_override_bbox.repeat(1, 4, 1)
        pred[:, :4, :][bbox_override_mask] = pred_coco[:, :4, :][bbox_override_mask]
        remove_non_face_logits_mask = should_override_bbox.bitwise_not().repeat(
            1, pred.shape[1] - 5, 1
        )
        pred[:, 5:, :][remove_non_face_logits_mask] = 0

        # TESTING CODE: Process predictions
        if True:
            from pathlib import Path
            from utils.general import scale_boxes, cv2, non_max_suppression
            from utils.plots import Annotator, colors

            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            print(pred[0].shape)

            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = "/home/ssun/Desktop/distill_test.jpg"
                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0
                names = t_coco.names
                annotator = Annotator(im0, line_width=3, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        c = int(cls)  # integer class
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()

                # Save results (image with detections)
                cv2.imwrite(save_path, im0)

        exit()
