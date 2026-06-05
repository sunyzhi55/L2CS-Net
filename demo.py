import argparse
import pathlib
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image, ImageOps
import json
import os

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()


def overlay_pitch_yaw(frame: np.ndarray, results, frame_idx, unit: str = 'rad') -> np.ndarray:
    """Overlay per-face pitch/yaw on the frame near each bbox.

    - unit='rad': show radians
    - unit='deg': show degrees
    """
    try:
        pitches = np.array(results.pitch)
        yaws = np.array(results.yaw)
        bboxes = np.array(results.bboxes)
    except Exception:
        return frame

    if pitches.size == 0 or yaws.size == 0:
        return frame

    n = min(len(pitches), len(yaws))
    if bboxes is not None and getattr(bboxes, 'ndim', 0) >= 2:
        n = min(n, len(bboxes))

    for fi in range(n):
        try:
            pitch = float(pitches[fi])
            yaw = float(yaws[fi])
        except Exception:
            continue

        if unit == 'deg':
            pitch_show = pitch * 180.0 / np.pi
            yaw_show = yaw * 180.0 / np.pi
            text = f"p={pitch_show:.1f}° y={yaw_show:.1f}°"
        else:
            text = f"p={pitch:.3f} y={yaw:.3f}"

        # default position (top-left)
        x, y = 10, 40 + 18 * fi

        # if bbox exists, place text near it
        try:
            x1, y1, x2, y2 = bboxes[fi].tolist()
            x = int(max(0, x1))
            y = int(max(0, y1) - 8)
        except Exception:
            pass

        cv2.putText(
            frame,
            f"[{frame_idx}-{fi}] {text}",
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.9,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return frame

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="5", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot. If empty, use default model in `models/`.', 
        default='', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--input', dest='input', help='Path to input video file. If not set, use webcam.',
        default="/data3/wangchangmiao/shenxy/Code/gaze/selfData/1.mp4", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path to save output video (rendered).',
        default="./output.mp4", type=str)
    parser.add_argument(
        '--results', dest='results', help='Path to save per-frame results (jsonl).',
        default=None, type=str)
    parser.add_argument(
        '--angles', dest='angles', help='Path to save per-frame pitch/yaw angles (CSV).',
        default="./angles.csv", type=str)
    parser.add_argument(
        '--no_window', default=True, help='Disable GUI window (headless). Do not call cv2.imshow().',
        action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    # choose snapshot: prefer provided snapshot (if it exists), else default model in `models/`
    if args.snapshot:
        candidate = pathlib.Path(args.snapshot)
        if candidate.exists():
            weights_path = candidate
        else:
            print(f"Warning: snapshot '{args.snapshot}' not found. Falling back to models/L2CSNet_gaze360.pkl")
            weights_path = CWD / 'models' / 'L2CSNet_gaze360.pkl'
    else:
        weights_path = CWD / 'models' / 'L2CSNet_gaze360.pkl'

    # final check: does weights_path exist? if not, raise informative error
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}. Provide a valid --snapshot path or place the model at this location.")

    gaze_pipeline = Pipeline(
        weights=weights_path,
        arch=arch,
        device = select_device(args.device, batch_size=1)
    )

    # open video source: file if provided, otherwise webcam
    if args.input:
        cap = cv2.VideoCapture(str(args.input))
    else:
        cap = cv2.VideoCapture(cam)

    # Check if the video source is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open video source (camera or input file)")

    # helper to serialize results into JSON-friendly formats
    def serialize_obj(obj):
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import torch as _torch
        except Exception:
            _torch = None

        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if _np is not None and isinstance(obj, _np.generic):
            return obj.item()
        if _np is not None and isinstance(obj, _np.ndarray):
            return obj.tolist()
        if _torch is not None and isinstance(obj, _torch.Tensor):
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return repr(obj)
        if isinstance(obj, dict):
            return {k: serialize_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [serialize_obj(v) for v in obj]
        try:
            return json.loads(json.dumps(obj))
        except Exception:
            return repr(obj)

    writer = None
    results_f = None
    angles_f = None
    frame_idx = 0

    # open results file if requested
    if args.results:
        results_f = open(args.results, 'w', encoding='utf-8')

    # open angles CSV if requested (save radians)
    if args.angles:
        angles_f = open(args.angles, 'w', encoding='utf-8')
        angles_f.write('frame_idx,face_idx,pitch_rad,yaw_rad\n')

    with torch.no_grad():
        while True:
            start_fps = time.time()

            success, frame = cap.read()
            if not success:
                # end of video file or camera read failure
                break

            # Process frame
            results = gaze_pipeline.step(frame)
            # 打印 results 的所有键值对（优先使用 serialize_obj 转为可序列化结构）
            try:
                serial = serialize_obj(results)
                # 美观输出
                print(json.dumps(serial, ensure_ascii=False, indent=2))
            except Exception:
                # 回退：如果是 dict，逐项打印；否则尝试打印可见属性
                try:
                    if isinstance(results, dict):
                        for k, v in results.items():
                            print(f"{k}: {v}")
                    else:
                        for attr in [a for a in dir(results) if not a.startswith('_')]:
                            try:
                                print(f"{attr}: {getattr(results, attr)}")
                            except Exception:
                                print(f"{attr}: <unreadable>")
                except Exception as e:
                    print(f"Error printing results: {e}")


            # Visualize output
            frame = render(frame, results)

            # Overlay pitch/yaw text for each detected face
            frame = overlay_pitch_yaw(frame, results, frame_idx, unit='rad')

            # initialize writer lazily so we know frame size and fps
            if args.output and writer is None:
                out_path = pathlib.Path(args.output)
                w = int(frame.shape[1])
                h = int(frame.shape[0])
                # determine fps
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0 or np.isnan(fps):
                    fps = 30.0
                # choose codec by extension
                ext = out_path.suffix.lower()
                if ext in ['.avi']:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

            # write rendered frame to output video if requested
            if writer is not None:
                writer.write(frame)

            # write results as one JSON object per line if requested
            if results_f is not None:
                entry = {'frame_idx': frame_idx, 'results': serialize_obj(results)}
                results_f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            # write pitch/yaw per detected face to CSV (radians)
            if angles_f is not None:
                try:
                    # results.pitch and results.yaw are numpy arrays in radians
                    pitches = np.array(results.pitch)
                    yaws = np.array(results.yaw)
                    # ensure same length
                    n = min(len(pitches), len(yaws))
                    for fi in range(n):
                        pitch_rad = float(pitches[fi])
                        yaw_rad = float(yaws[fi])
                        angles_f.write(f"{frame_idx},{fi},{pitch_rad:.6f},{yaw_rad:.6f}\n")
                except Exception:
                    # if results has no faces or unexpected shape, skip
                    pass

            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            if not args.no_window:
                cv2.imshow("Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    # cleanup
    if writer is not None:
        writer.release()
    if results_f is not None:
        results_f.close()
    if angles_f is not None:
        angles_f.close()
    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    

"""

python3 demo.py --device "0" --results ./results.jsonl --no_window

results = gaze_pipeline.step(frame)
results格式:
GazeResultContainer(
pitch=array([0.01819312], dtype=float32), 
yaw=array([0.07074182], dtype=float32), 
bboxes=array([[375.4524 , 388.37125, 800.92236, 932.9131 ]], dtype=float32),
landmarks=array([[[517.5697 , 592.8084 ],\n        
 [702.82   , 584.5797 ],\n        [634.3065 , 705.5309 ],\n       
   [543.1326 , 804.06537],\n        [689.6688 , 796.97076]]], dtype=float32), 
scores=array([0.99994624], dtype=float32))"

pitch：
含义：模型预测的俯仰角（pitch）。
类型/单位/形状：NumPy 数组，单位为 弧度（radians）。通常形状为 (N,)（或在无脸时可能是 (0,1)），N 为该帧检测到的人脸数。
yaw：
含义：模型预测的偏航角（yaw）。
类型/单位/形状：NumPy 数组，单位为 弧度，形状与 pitch 对应，每个索引 i 表示同一张脸的 yaw 值。
bboxes：
含义：人脸检测得到的边界框（bounding box）。
格式/坐标系：像素坐标，通常为 [x_min, y_min, x_max, y_max]（左上和右下角），相对于输入帧的原始坐标系。
形状：(N, 4)。（每行一个框）
landmarks：
含义：检测器返回的人脸关键点（如左右眼、鼻子、嘴角等）。
格式/坐标系：通常为每脸若干点的像素坐标，可能的形状为 (N,5,2)（5 个点，每个点 (x,y)）或 (N,10)（扁平化）。确切布局取决于 RetinaFace 实现（一般点顺序：左眼、右眼、鼻子、左嘴角、右嘴角）。
scores：
含义：检测器对每个 bounding box 的置信度分数（confidence）。
类型/形状：一维数组 (N,)，float，范围通常在 0..1 之间。

"""