import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import csv
import cv2
import numpy as np
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.plots import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # プロジェクトルートディレクトリ

# 入出力パスの汎用化（GitHub公開用）
input_folder = ROOT / "runs/detect/exp"
output_folder = ROOT / "output"
output_csv_path = output_folder / "hirame_areas.csv"

# 関数: 相対座標を絶対座標に変換
def relative_to_absolute(x_center, y_center, width, height, img_width, img_height):
    abs_x = x_center * img_width
    abs_y = y_center * img_height
    abs_width = width * img_width
    abs_height = height * img_height
    top_left_x = abs_x - abs_width / 2
    top_left_y = abs_y - abs_height / 2
    return top_left_x, top_left_y, abs_width, abs_height

# CSVファイルに書き込む関数を定義
def write_to_csv(image_name, cls, conf, xywh):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        # ヘッダーを書き込む
        if f.tell() == 0:
            writer.writerow(["image", "class", "confidence", "x_center", "y_center", "width", "height"])
        writer.writerow([image_name, cls, conf, *xywh])

# print_args 関数を定義
def print_args(args):
    for k, v in args.items():
        print(f"{k}: {v}")

@smart_inference_mode()
def run(
    weights=ROOT / "model.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=True,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            s += "%gx%g " % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_csv:  # Write to CSV file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        write_to_csv(p.name, names[int(cls)], conf.item(), xywh)

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    # 面積の計算とCSVファイルへの保存
    if os.path.exists(csv_path):
        detections_df = pd.read_csv(csv_path)

        output_data = []

        for image_file in os.listdir(input_folder):
            if image_file.endswith('.JPEG'):
                image_path = os.path.join(input_folder, image_file)

                # 画像の読み込み
                image = cv2.imread(image_path)
                image_width, image_height = image.shape[1], image.shape[0]

                # 該当する画像に対する検出データを取得
                img_detections = detections_df[detections_df['image'] == image_file]
                if img_detections.empty:
                    continue

                # standardとflounderの座標を取得
                try:
                    standard_row = img_detections[img_detections['class'] == 'standard'].iloc[0]
                    flounder_row = img_detections[img_detections['class'] == 'flounder'].iloc[0]
                except IndexError:
                    continue

                standard_top_left_x, standard_top_left_y, standard_width, standard_height = relative_to_absolute(
                    standard_row['x_center'], standard_row['y_center'], standard_row['width'], standard_row['height'], image_width, image_height)
                flounder_top_left_x, flounder_top_left_y, flounder_width, flounder_height = relative_to_absolute(
                    flounder_row['x_center'], flounder_row['y_center'], flounder_row['width'], flounder_row['height'], image_width, image_height)

                standard_top_left = (int(standard_top_left_x), int(standard_top_left_y))
                standard_bottom_right = (int(standard_top_left_x + standard_width), int(standard_top_left_y + standard_height))
                flounder_top_left = (int(flounder_top_left_x), int(flounder_top_left_y))
                flounder_bottom_right = (int(flounder_top_left_x + flounder_width), int(flounder_top_left_y + flounder_height))

                # ヒラメの輪郭検出（枠を除外）
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 枠を除外するためのマスク処理
                mask = np.zeros_like(binary_image)
                cv2.rectangle(mask, (flounder_top_left[0] + 5, flounder_top_left[1] + 5), (flounder_bottom_right[0] - 5, flounder_bottom_right[1] - 5), 255, -1)

                # マスクを適用して枠を除外
                masked_binary_image = cv2.bitwise_and(binary_image, mask)
                flounder_roi = masked_binary_image[flounder_top_left[1]:flounder_bottom_right[1], flounder_top_left[0]:flounder_bottom_right[0]]

                # ノイズ除去のためにモルフォロジー変換を追加
                kernel = np.ones((5, 5), np.uint8)
                cleaned_flounder_roi = cv2.morphologyEx(flounder_roi, cv2.MORPH_CLOSE, kernel)

                # 輪郭検出とノイズ除去
                contours_flounder, _ = cv2.findContours(cleaned_flounder_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_flounder = [cnt for cnt in contours_flounder if cv2.contourArea(cnt) > 500]  # 小さなノイズを除外
                if contours_flounder:
                    largest_contour_flounder = max(contours_flounder, key=cv2.contourArea)
                    contours_flounder = [largest_contour_flounder]
                    cv2.drawContours(image, contours_flounder, -1, (0, 255, 0), 2, offset=(flounder_top_left[0], flounder_top_left[1]))
                    flounder_area_px = cv2.contourArea(largest_contour_flounder)
                else:
                    flounder_area_px = 0

                # 黒い四角の輪郭検出
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 30])
                mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
                mask_black_full = np.zeros_like(mask_black)
                mask_black_full[standard_top_left[1]:standard_bottom_right[1], standard_top_left[0]:standard_bottom_right[0]] = \
                    mask_black[standard_top_left[1]:standard_bottom_right[1], standard_top_left[0]:standard_bottom_right[0]]
                contours_black, _ = cv2.findContours(mask_black_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours_black:
                    largest_contour_black = max(contours_black, key=cv2.contourArea)
                    contours_black = [largest_contour_black]
                    cv2.drawContours(image, contours_black, -1, (255, 0, 0), 2)
                    black_square_area_px = cv2.contourArea(largest_contour_black)
                else:
                    black_square_area_px = 0

                # 面積計算
                flounder_area_cm2 = flounder_area_px / black_square_area_px if black_square_area_px > 0 else 0

                cv2.drawContours(image, contours_black, -1, (255, 0, 0), 2)
                cv2.drawContours(image, contours_flounder, -1, (0, 255, 0), 2, offset=(flounder_top_left[0], flounder_top_left[1]))

                # 画像保存
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                result_image_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.JPEG')
                cv2.imwrite(result_image_path, image)

                # 面積結果をデータリストに追加
                output_data.append({'image': os.path.splitext(image_file)[0] + '.JPEG', 'flounder_area_cm2': flounder_area_cm2})

        # 結果をデータフレームに変換してCSVに保存
        output_df = pd.DataFrame(output_data, columns=['image', 'flounder_area_cm2'])
        output_df.to_csv(output_csv_path, index=False)

        print(f"処理が完了しました。結果のCSVファイルと画像は {output_folder} に保存されています。")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "model.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)



