import os
import shutil
import glob

def copy_binvox_files_only(original_root, new_root):
    for class_dir in os.listdir(original_root):
        class_path = os.path.join(original_root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.isdir(split_path):
                continue

            # 出力先のディレクトリを作成
            new_split_path = os.path.join(new_root, class_dir, split)
            os.makedirs(new_split_path, exist_ok=True)

            # .binvox ファイルだけをコピー
            for file_path in glob.glob(os.path.join(split_path, '*.binvox')):
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(new_split_path, file_name)
                shutil.copyfile(file_path, dest_path)

    print(f"Copied all .binvox files from {original_root} to {new_root}.")

# 使用例
original_data_root = './data/ModelNet10'
new_data_root = './data/ModelNet10_binvox_only'
copy_binvox_files_only(original_data_root, new_data_root)
