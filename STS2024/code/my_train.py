import os
import shutil
import random
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
from ultralytics import YOLO


# 将 NIfTI 文件转换为 PNG 并保存
def nii_to_png(nii_file, output_folder, start_index=0):
    # 加载 NIfTI 文件
    img = nib.load(nii_file)
    data = img.get_fdata()

    # 获取切片的数量
    num_slices = data.shape[2]

    # 遍历每个切片并保存为 PNG
    for i in range(num_slices):
        slice_data = data[:, :, i]

        # 将切片数据归一化到0-255范围
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
        slice_data = slice_data.astype(np.uint8)

        # 生成文件名，保证长度一致，序号从0开始
        filename = f"{start_index + i:05d}.png"
        filepath = os.path.join(output_folder, filename)

        # 保存为 PNG 文件
        img = Image.fromarray(slice_data)
        img.save(filepath)

    # 返回下一个文件应该从哪个编号开始
    return start_index + num_slices

# 处理 NIfTI 文件并将其切片保存到指定文件夹
def process_nii_files(input_dir, base_output_dir, start_index=0):
    # 创建输出文件夹（如果不存在）
    os.makedirs(base_output_dir, exist_ok=True)

    # 获取文件夹中的所有 NIfTI 文件
    nii_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # 遍历并处理每个 NIfTI 文件
    for nii_file in nii_files:
        nii_file_path = os.path.join(input_dir, nii_file)
        print(f"正在处理: {nii_file_path}")
        start_index = nii_to_png(nii_file_path, base_output_dir, start_index)

    print(f"处理完成，所有 NIfTI 文件已保存为 PNG。")


def generate_yolo_txt(slice_data, txt_path):
    height, width = slice_data.shape

    # 获取唯一标签（排除背景，假设背景为0）
    unique_labels = np.unique(slice_data)
    unique_labels = unique_labels[unique_labels > 0]

    with open(txt_path, 'w') as txt_file:
        for label in unique_labels:
            # 创建掩码
            mask = (slice_data == label).astype(np.uint8)

            # 寻找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 将轮廓转为YOLO格式
                yolo_format = []
                for point in contour:
                    x, y = point[0]
                    x_norm = x / width
                    y_norm = y / height
                    yolo_format.extend([x_norm, y_norm])

                # 写入YOLO格式：类别ID和归一化坐标
                txt_file.write(f'{int(label)} ' + ' '.join(map(str, yolo_format)) + '\n')

        print(f'Generated {txt_path}')



# 处理文件夹中的所有 NIfTI 文件，生成 YOLO 标签文件
def save_cbct_labels_to_yolo_format(input_folder, labels_folder):
    # 确保标签文件夹存在
    os.makedirs(labels_folder, exist_ok=True)

    # 获取文件夹内所有的 NIfTI 文件
    nii_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # 初始化切片计数
    slice_count = 0

    # 遍历每个 NIfTI 文件
    for nii_file in nii_files:
        # 读取 NIfTI 文件
        nii_img = nib.load(os.path.join(input_folder, nii_file))
        data = nii_img.get_fdata()

        # 遍历每一层（沿 z 轴方向）
        for i in range(data.shape[2]):
            # 提取切片数据
            slice_data = data[:, :, i]

            # 生成文件名
            label_filename = f'{slice_count:05d}.txt'

            # 生成YOLO格式的标签文件
            generate_yolo_txt(slice_data, os.path.join(labels_folder, label_filename))

            # 增加切片计数
            slice_count += 1

    print(f"处理完成，共生成 {slice_count} 个标签文件。")


def filter_dataset(img_dir, txt_dir):
    # 获取所有的txt文件
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])

    # 逐个检查txt文件
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        with open(txt_path, 'r') as file:
            lines = file.readlines()

            # 检查是否为空和每行是否至少有七个数值
            if lines and all(len(line.split()) < 7 for line in lines):
                # 找到对应的图片
                img_file = txt_file.replace('.txt', '.png')
                image_path = os.path.join(img_dir, img_file)
                os.remove(image_path)
                os.remove(txt_path)



# 定义数据划分函数
def split_dataset(base_dir, output_dir, train_ratio):
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')

    # 获取所有nii文件
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # 确保图像和掩码文件数量一致
    assert len(image_files) == len(mask_files), "图像和掩码文件数量不一致！"

    # 随机打乱数据
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files[:], mask_files[:] = zip(*combined)

    # 划分训练集和验证集
    train_size = int(train_ratio * len(image_files))
    train_images, val_images = image_files[:train_size], image_files[train_size:]
    train_masks, val_masks = mask_files[:train_size], mask_files[train_size:]

    # 定义训练集和验证集的路径
    train_images_dir = os.path.join(output_dir, 'train1', 'images')
    train_masks_dir = os.path.join(output_dir, 'train1', 'masks')
    val_images_dir = os.path.join(output_dir, 'val1', 'images')
    val_masks_dir = os.path.join(output_dir, 'val1', 'masks')

    # 创建相关目录
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)

    # 复制训练集数据
    for img_file, mask_file in zip(train_images, train_masks):
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(train_images_dir, img_file))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(train_masks_dir, mask_file))

    # 复制验证集数据
    for img_file, mask_file in zip(val_images, val_masks):
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(val_images_dir, img_file))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(val_masks_dir, mask_file))

    print(f"数据集划分完成：训练集 {len(train_images)} 个，验证集 {len(val_images)} 个。")


def filter_dataset(img_dir, txt_dir, output_img_dir, output_txt_dir):
    # 创建保存筛选后的图片和txt文件的目录
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    # 获取所有的图片和txt文件
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])

    # 逐个检查txt文件
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        with open(txt_path, 'r') as file:
            lines = file.readlines()

            # 检查是否为空和每行是否至少有七个数值
            if lines and all(len(line.split()) >= 7 for line in lines):
                # 找到对应的图片
                img_file = txt_file.replace('.txt', '.png')
                if not os.path.exists(os.path.join(img_dir, img_file)):
                    img_file = txt_file.replace('.txt', '.png')

                if img_file in images:
                    # 复制文件到新的目录
                    shutil.copy(os.path.join(img_dir, img_file), os.path.join(output_img_dir, img_file))
                    shutil.copy(txt_path, os.path.join(output_txt_dir, txt_file))
                    print(f'筛选通过: {img_file} 和 {txt_file}')




def split_and_process_nii_files(input_dir, output_dir, train_ratio):
    # 划分数据集
    split_dataset(input_dir, output_dir, train_ratio)

    # 定义文件夹路径
    train_images_dir = os.path.join(output_dir, 'train1', 'images')
    train_masks_dir = os.path.join(output_dir, 'train1', 'masks')
    val_images_dir = os.path.join(output_dir, 'val1', 'images')
    val_masks_dir = os.path.join(output_dir, 'val1', 'masks')

    train_images_dir_filter = os.path.join(output_dir, 'train', 'images')
    train_masks_dir_filter = os.path.join(output_dir, 'train', 'labels')
    val_images_dir_filter = os.path.join(output_dir, 'val', 'images')
    val_masks_dir_filter = os.path.join(output_dir, 'val', 'labels')

    # images到切片
    process_nii_files(train_images_dir, train_images_dir)
    process_nii_files(val_images_dir, val_images_dir)

    # 在处理完成后再删除 NIfTI 文件
    for file in os.listdir(train_images_dir):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            os.remove(os.path.join(train_images_dir, file))

    for file in os.listdir(val_images_dir):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            os.remove(os.path.join(val_images_dir, file))

    # masks到yolo格式
    save_cbct_labels_to_yolo_format(train_masks_dir, train_masks_dir)
    save_cbct_labels_to_yolo_format(val_masks_dir, val_masks_dir)

    # 处理完标签后再删除 NIfTI 文件
    for file in os.listdir(train_masks_dir):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            os.remove(os.path.join(train_masks_dir, file))

    for file in os.listdir(val_masks_dir):
        if file.endswith('.nii') or file.endswith('.nii.gz'):
            os.remove(os.path.join(val_masks_dir, file))

    # 过滤数据集
    filter_dataset(train_images_dir, train_masks_dir, train_images_dir_filter, train_masks_dir_filter)
    filter_dataset(val_images_dir, val_masks_dir, val_images_dir_filter, val_masks_dir_filter)
    del_train1 = os.path.join(input_dir, 'train1')
    del_val1 = os.path.join(input_dir, 'val1')
    shutil.rmtree(del_train1)
    shutil.rmtree(del_val1)

# 使用示例
if __name__ == "__main__":

    # 包含 images 和 masks 文件夹的上一级目录
    base_dir = r'D:\PyCharm_code\STS_2024\STS2024\code\data'
    # 输出文件夹路径，train 和 val 将保存在这里
    output_dir = base_dir
    #数据预处理
    split_and_process_nii_files(base_dir, output_dir, 0.8)
    # 加载模型
    model = YOLO("D:\PyCharm_code\STS_2024\STS2024\code\yolov8-seg.yaml")
    model = YOLO("D:\PyCharm_code\STS_2024\STS2024\code\yolov8n-seg.pt")
    # 训练模型
    model.train(data='D:\PyCharm_code\STS_2024\STS2024\code\coco128-seg.yaml', epochs=10,
                imgsz=640)

