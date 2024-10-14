import os
import shutil

import cv2
import nibabel as nib
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO



def get_nii_image_size(nii_file):
    nii_data = nib.load(nii_file)
    return nii_data.shape[0:2]


def generate_masks_from_detections(results, image_size):
    masks = []
    for r in results:
        masks.append(create_mask_from_detections(r, image_size))
    return masks


def create_mask_from_detections(detections, image_size):
    mask = np.zeros(image_size, dtype=np.uint8)

    for detection in detections:
        class_id = int(detection.boxes[0].cls)  # 获取第一个框的类别
        coords_array = detection.masks.xyn[0]  # 获取坐标
        contour = []

        for coords in coords_array:
            x = int(coords[0] * (image_size[1] - 1))
            y = int(coords[1] * (image_size[0] - 1))
            contour.append([x, y])

        if len(contour) > 0:
            contour = np.array(contour, dtype=np.int32)
            contour = contour.reshape((-1, 1, 2))
            cv2.drawContours(mask, [contour], -1, color=class_id, thickness=cv2.FILLED)

    return mask


def recompose_nii_from_slices(mask_folder, output_nii_file, original_affine, original_header, num_slices):
    slice_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    mask_img = Image.open(os.path.join(mask_folder, slice_files[0]))
    width, height = mask_img.size

    volume = np.zeros((height, width, num_slices), dtype=np.uint8)

    for i, slice_file in enumerate(slice_files):
        img = Image.open(os.path.join(mask_folder, slice_file)).convert('L')
        volume[:, :, i] = np.array(img)

    nii_img = nib.Nifti1Image(volume, original_affine, original_header)
    nib.save(nii_img, output_nii_file)
    print(f"Saved NIfTI file: {output_nii_file}")


def my_predict():
    model_path = r"D:\PyCharm_code\STS_2024\STS2024\code\runs\segment\train2\weights\best.pt"
    # config device
    device = torch.device('cuda:0')
    # load model and checkpoint file
    yolo_model = YOLO(model_path).to(device).float()
    temp_dir = os.path.join(OUTPUT_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    with torch.no_grad():

        # load the current case since there would only be one case in the INPUT_DIR

        case_name = os.listdir(INPUT_DIR)[0]
        # image pre-processing
        nii_file = os.path.join(INPUT_DIR, case_name)

        case_tag = case_name.split('.')[0]
        # 临时文件夹存储切片和mask
        output_folder = os.path.join(temp_dir, "slice")
        os.makedirs(output_folder, exist_ok=True)

        mask_output_folder = os.path.join(temp_dir, "mask_images")
        os.makedirs(mask_output_folder, exist_ok=True)

        image_size = get_nii_image_size(nii_file)
        print(f'Processing NIfTI file: {case_name} with image size: {image_size}')

        img = nib.load(nii_file)
        data = img.get_fdata()
        original_affine = img.affine
        original_header = img.header

        num_slices = data.shape[2]
        for i in range(num_slices):
            slice_data = data[:, :, i]
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
            slice_data = slice_data.astype(np.uint8)

            filename = f"{i + 1:04d}.jpg"
            filepath = os.path.join(output_folder, filename)
            img_slice = Image.fromarray(slice_data)
            img_slice.save(filepath)

            results = yolo_model(filepath)
            masks = generate_masks_from_detections(results, image_size)
            for j, mask in enumerate(masks):
                mask_filename = f'{i + 1:04d}.png'
                mask_filepath = os.path.join(mask_output_folder, mask_filename)
                cv2.imwrite(mask_filepath, mask)
                print(f'Saved mask: {mask_filepath}')

        case_tag = case_name.split('.')[0]

        output_nii_file = os.path.join(OUTPUT_DIR, '%s_Mask.nii.gz' % case_tag.replace('STS24_', ''))
        recompose_nii_from_slices(mask_output_folder, output_nii_file, original_affine, original_header, num_slices)

        # 删除中间的mask_images和slice文件夹

        shutil.rmtree(temp_dir)



if __name__ == "__main__":

    INPUT_DIR = 'D:\PyCharm_code\STS_2024\STS2024\code\input'
    OUTPUT_DIR = 'D:\PyCharm_code\STS_2024\STS2024\code\output'

    my_predict()
