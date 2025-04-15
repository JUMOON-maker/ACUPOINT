"""
改代码用来将图片文件按次序命名
"""
import os


# 定义 train 文件夹的路径
train_folder = 'train'

# 检查 train_folder 是否存在
if os.path.exists(train_folder) and os.path.isdir(train_folder):
    # 获取 train 文件夹下的所有子文件夹
    sub_folders = [f for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, f))]
    # 对获取到的子文件夹列表进行排序，确保顺序一致
    sub_folders.sort()

    # 找出所有已存在的 face 命名文件夹
    existing_face_folders = [folder for folder in sub_folders if folder.startswith('face')]
    existing_face_numbers = set(int(folder[4:]) for folder in existing_face_folders if folder[4:].isdigit())

    # 重命名子文件夹
    new_name_mapping = {}
    for i, sub_folder in enumerate(sub_folders, start = 1):
        new_name = f'face{i}'
        while i in existing_face_numbers:
            i += 1
            new_name = f'face{i}'
        new_name_mapping[sub_folder] = new_name
        existing_face_numbers.add(i)

    for sub_folder, new_name in new_name_mapping.items():
        old_path = os.path.join(train_folder, sub_folder)
        new_path = os.path.join(train_folder, new_name)
        try:
            os.rename(old_path, new_path)
            print(f"已将 {old_path} 重命名为 {new_path}")
        except Exception as e:
            print(f"重命名 {old_path} 时出错: {e}")
else:
    print(f"指定的 train 文件夹 {train_folder} 不存在。")

