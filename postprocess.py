import os
import cv2
import numpy as np
import argparse

def crop_center_from_folder(input_folder, output_folder, target_size=(428, 240)):
    """從資料夾中的圖像中心切割指定大小的區域"""
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 讀取輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 讀取圖片
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # 計算切割後的範圍
            y_center = image.shape[0] // 2
            x_center = image.shape[1] // 2
            y_start = y_center - target_size[1] // 2
            y_end = y_center + target_size[1] // 2
            x_start = x_center - target_size[0] // 2
            x_end = x_center + target_size[0] // 2

            # 切割圖片
            cropped_image = image[y_start:y_end, x_start:x_end]

            # 儲存結果
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', required=True, help='輸入資料夾路徑')
    parser.add_argument('--target_folder', required=True, help='輸出資料夾路徑')
    args = parser.parse_args()

    input_folder = args.source_folder
    output_folder = args.target_folder

    crop_center_from_folder(input_folder, output_folder)

if __name__ == '__main__':
    main()