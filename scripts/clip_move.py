"""
    拷贝文件夹并将其重命名为：[id]_[difficulty]_[task] 格式，
"""


from pathlib import Path
import shutil

def main():
    src_root = Path("./20260818_clip")
    dst_root = Path("./20260821_clip")

    # 若需要自动覆盖旧输出，取消下面两行注释
    # if dst_root.exists():
    #     shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    for difficulty in ("easy", "hard"):
        diff_dir = src_root / difficulty
        if not diff_dir.is_dir():
            continue

        for id_dir in diff_dir.iterdir():
            if not id_dir.is_dir():
                continue
            folder_id = id_dir.name

            for task_dir in id_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task = task_dir.name

                new_dir_name = f"{folder_id}_{difficulty}_{task}"
                dst_task_dir = dst_root / new_dir_name

                print(f"Copying: {task_dir} --> {dst_task_dir}")
                shutil.copytree(task_dir, dst_task_dir)

    print(f"\n✅ All directories copied to {dst_root.resolve()}")

if __name__ == "__main__":
    main()