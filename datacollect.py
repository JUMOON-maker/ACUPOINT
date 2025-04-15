"""
此代码是用来将已经标记好的图片进行处理并收集数据
"""
import os
import cv2
import numpy as np


"""
将图片调整为128x128
"""
def resize_image_with_padding(image, target_size=(1024, 1024)):
    # 获取原始图像的高度和宽度
    height, width = image.shape[:2]
    # 计算目标大小的宽度和高度
    target_width, target_height = target_size
    # 计算缩放比例，取宽度和高度缩放比例中的较小值
    scale = min(target_width / width, target_height / height)
    # 计算缩放后的宽度和高度
    new_width = int(width * scale)
    new_height = int(height * scale)
    # 对原始图像进行等比例缩放
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # 创建一个空白的目标大小的图像，背景颜色为黑色
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    # 计算在空白图像中放置缩放后图像的起始位置，使其居中
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    # 将缩放后的图像放置到空白图像的指定位置
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return padded_image


data_dir = 'train'  # 数据集文件夹路径
"""
遍历数据集文件夹中的每个子文件夹
读取其中的图片数据
"""
for sub_dir in os.listdir(data_dir):
    sub_dir_path = os.path.join(data_dir, sub_dir)  # 拼接当前子文件夹的完整路径
    if os.path.isdir(sub_dir_path):  # 检查当前路径是否为一个文件夹
        # 遍历子文件夹中的图片文件
        image_files = [f for f in os.listdir(sub_dir_path) if f.endswith(('.jpg', '.png', '.jpeg'))]


        def custom_sort_key(file_path):
            # 提取文件名中的数字部分
            file_name = os.path.basename(file_path)
            number = int(os.path.splitext(file_name)[0])
            return number


        # 按自定义规则排序图片文件(图片文件名为1、2、3。。，按照数字大小进行排序)
        image_files.sort(key=custom_sort_key)

        for image_file in image_files:
            # 提取图片文件名中的数字部分
            number = custom_sort_key(os.path.join(sub_dir_path, image_file))
            # 构建新的 txt 文件名
            new_txt_file_name = f"{number}.txt"
            # 拼接完整的 txt 文件路径
            txt_file_path = os.path.join(sub_dir_path, new_txt_file_name)

            with open(txt_file_path, 'w') as txt_file:  # 打开该文本文件（对应该图片）
                image_path = os.path.join(sub_dir_path, image_file)  # 拼接每一张图片路径
                image = cv2.imread(image_path)  # 读取图像
                if image is not None:
                    resized_image = resize_image_with_padding(image, target_size=(1024, 1024))  # 调整图像大小为 128x128
                    # 找出红色像素的掩码
                    red_mask = np.where((resized_image[:, :, 2] > 190) & (resized_image[:, :, 1] < 70) & (
                                resized_image[:, :, 0] < 70), 255, 0).astype(np.uint8)
                    # 进行连通区域分析
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask)
                    print(f"这是文件{sub_dir}")
                    print(f"图片 {image_file} 检测到的坐标数量: {num_labels - 1}")
                    # 跳过背景（标签为 0）
                    for i in range(1, num_labels):
                        centroid_x, centroid_y = centroids[i]
                        centroid_x = int(centroid_x)
                        centroid_y = int(centroid_y)
                        txt_file.write(f'{centroid_x} {centroid_y}\n')