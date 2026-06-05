#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批处理 FatigueGuard 数据集
"""
import json, numpy as np, cv2, argparse
from pathlib import Path
from FatigueGuard_preprocess_single import GazeToPoint
from gaze_tracking.model import EyeModel

def load_easy_targets(txt_path):
    targets = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "," in line:
                x, y = line.split(",")
            else:
                parts = line.split()
                if len(parts) < 2: continue
                x, y = parts[0], parts[1]
            targets.append((int(float(x)), int(float(y))))
    return targets

def load_hard_targets(npy_path):
    return list(np.load(npy_path, allow_pickle=True))

def augment_easy_targets(targets, total_frames, fps=30):
    frames_per_target = fps * 3
    augmented = []
    for t in targets:
        augmented.extend([t] * frames_per_target)
    if len(augmented) < total_frames:
        augmented.extend([targets[-1]] * (total_frames - len(augmented)))
    return augmented[:total_frames]

def augment_hard_targets(targets, total_frames):
    frames_per_index = 180
    augmented = []
    for item in targets:
        augmented.extend([item['centers']] * frames_per_index)
    if len(augmented) < total_frames:
        augmented.extend([targets[-1]['centers']] * (total_frames - len(augmented)))
    return augmented[:total_frames]

def find_calibration_files(subject_dir, state, difficulty):
    candidates = [
        (subject_dir / "results", subject_dir / "STransG"),
        (subject_dir / state / difficulty / "results", subject_dir / state / difficulty / "STransG"),
        (subject_dir / state / "results", subject_dir / state / "STransG"),
    ]
    for result_dir, strans_dir in candidates:
        if result_dir.exists() and strans_dir.exists():
            required = ["STransG.npy", "StG.npy", "scaleWtG.npy", "STransW.npy", "StW.npy"]
            if all((strans_dir / f).exists() for f in required):
                return result_dir, strans_dir
    return None, None

def process_video(video_path, target_data, task_type, label, subject_id, output_dir, result_dir, strans_dir, args):
    try:
        temp_jsonl = output_dir / f"temp_{subject_id}_{task_type}_{label}.jsonl"
        args.input, args.jsonl, args.directory = str(video_path), str(temp_jsonl), str(result_dir.parent)
        args.stg_npy = str(strans_dir / "STransG.npy")
        args.stw_npy = str(strans_dir / "STransW.npy")
        args.scale_wtg = str(strans_dir / "scaleWtG.npy")
        args.stg_aux_npy = str(strans_dir / "StG.npy")
        args.stw_aux_npy = str(strans_dir / "StW.npy")
        
        print(f"  处理视频: {video_path.name}")
        gaze_to_point = GazeToPoint(result_dir.parent, args)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  错误: 无法打开视频")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"  视频: {total_frames} 帧, {fps} fps")
        
        gaze_to_point.RunGazeOnScreen(model=EyeModel("./"), cap=cap, sfm=(args.mode == "sfm"))
        cap.release()
        
        if task_type == 'easy':
            augmented_targets = augment_easy_targets(target_data, total_frames, fps)
        else:
            augmented_targets = augment_hard_targets(target_data, total_frames)
        
        output_jsonl = output_dir / f"{subject_id}_{task_type}_{label}.jsonl"
        with open(temp_jsonl) as f_in, open(output_jsonl, 'w') as f_out:
            for idx, line in enumerate(f_in):
                if idx >= len(augmented_targets): break
                data = json.loads(line.strip())
                if task_type == 'easy':
                    data['target_xy_px'] = list(augmented_targets[idx])
                else:
                    data['target_centers_xy_px'] = [list(c) for c in augmented_targets[idx]]
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        temp_jsonl.unlink()
        print(f"  完成: {output_jsonl.name}")
        return True
    except Exception as e:
        print(f"  错误: {e}")
        return False

def process_subject(subject_dir, output_dir, base_args):
    subject_id = subject_dir.name
    print(f"\n处理受试者: {subject_id}")
    results = {}
    
    for state in ['alert', 'sleepy']:
        state_dir = subject_dir / state
        if not state_dir.exists(): continue
        
        for difficulty in ['easy', 'hard']:
            task_dir = state_dir / difficulty
            if not task_dir.exists(): continue
            
            video_path = task_dir / "training_video.mp4"
            if not video_path.exists(): continue
            
            result_dir, strans_dir = find_calibration_files(subject_dir, state, difficulty)
            if not result_dir:
                print(f"  跳过: 找不到校准文件")
                continue
            
            if difficulty == 'easy':
                target_file = task_dir / "centers_easy.txt"
                if not target_file.exists(): continue
                target_data = load_easy_targets(str(target_file))
            else:
                target_file = task_dir / "Gaze_hard_centers.npy"
                if not target_file.exists(): continue
                target_data = load_hard_targets(str(target_file))
            
            key = f"{state}_{difficulty}"
            results[key] = process_video(
                video_path, target_data, difficulty, state, subject_id,
                output_dir, result_dir, strans_dir, base_args
            )
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data3/wangchangmiao/shenxy/Code/gaze/Data2", help="数据集根目录")
    parser.add_argument("--output_dir", default="/data3/wangchangmiao/shenxy/Code/gaze/DataOutput", help="输出目录")
    parser.add_argument("--device", default="cpu", help="设备")
    parser.add_argument("--weights", default="models/L2CSNet_gaze360.pkl")
    parser.add_argument("--arch", default="ResNet50")
    parser.add_argument("--mode", default="sfm", choices=["global", "sfm"])
    parser.add_argument("--sfm_openvino_device", default="CPU")
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--subjects", default=None, help="指定受试者ID，用逗号分隔，如 01,02,03")
    parser.add_argument("--camera_data_dir", default="./camera_data", help="相机标定数据目录路径，默认为 ./camera_data")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.subjects:
        subject_ids = [s.strip() for s in args.subjects.split(',')]
        subject_dirs = [data_root / sid for sid in subject_ids if (data_root / sid).exists()]
    else:
        subject_dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    
    print(f"共找到 {len(subject_dirs)} 个受试者")
    
    for subject_dir in subject_dirs:
        results = process_subject(subject_dir, output_dir, args)
    
    print("\n完成!")

if __name__ == '__main__':
    main()
