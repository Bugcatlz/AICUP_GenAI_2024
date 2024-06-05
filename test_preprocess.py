import os
import cv2
import shutil
import argparse
import numpy as np
from multiprocessing import Pool

def add_border_to_folder(border_size, folder):
    """為資料夾中的圖像添加邊框"""
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            top_border = (border_size - image.shape[0]) // 2
            bottom_border = border_size - image.shape[0] - top_border
            left_border = (border_size - image.shape[1]) // 2
            right_border = border_size - image.shape[1] - left_border

            image_with_border = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=(0, 0, 255))

            output_path = os.path.join(folder, filename)
            cv2.imwrite(output_path, image_with_border)

def move_files(source_folder, river_folder, road_folder):
    """將源資料夾中的檔案移動到 river 和 road 資料夾"""
    os.makedirs(river_folder, exist_ok=True)
    os.makedirs(road_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if "RI" in filename and "RO" not in filename:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(river_folder, filename))
        elif "RO" in filename:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(road_folder, filename))

def modified_data_type(folder):
    """將資料夾中的標籤檔案名稱從 .png 修改為 .jpg"""
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            old_file_path = os.path.join(folder, filename)
            new_file_path = os.path.join(folder, filename.replace('.png', '.jpg'))
            os.rename(old_file_path, new_file_path)

def resize_image(img, target_size=(32, 32)):
    """將輸入圖像調整為目標尺寸"""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def l1_distance(imgA, imgB):
    """計算兩個灰階圖像之間的 L1 距離"""
    return np.sum(np.abs(imgA.astype(np.float32) - imgB.astype(np.float32)))

def find_most_similar_image(args):
    img_path, img_label_dir = args

    imgA = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imgA = resize_image(imgA, (32, 32))

    img_label_paths = [os.path.join(img_label_dir, img_name) for img_name in os.listdir(img_label_dir)]
    imgBs = [resize_image(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (32, 32)) for path in img_label_paths]

    distances = [l1_distance(imgA, img) for img in imgBs]

    min_distance = min(distances)
    most_similar_image_index = distances.index(min_distance)
    most_similar_image_path = img_label_paths[most_similar_image_index]

    return most_similar_image_path, min_distance

def process_images(source_dir, img_label_dir, target_dir):
    """處理圖像並將最相似的圖像複製到目標目錄"""
    os.makedirs(target_dir, exist_ok=True)

    source_files = sorted(os.listdir(source_dir))
    args = [(os.path.join(source_dir, img_name), img_label_dir) for img_name in source_files]

    with Pool() as pool:
        results = pool.map(find_most_similar_image, args)

    for imgA_path, result in zip(args, results):
        most_similar_image_path, min_distance = result
        target_path = os.path.join(target_dir, os.path.basename(imgA_path[0]))
        shutil.copy(most_similar_image_path, target_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', required=True, help='源資料夾路徑')
    parser.add_argument('--target_folder', required=True, help='目標資料夾路徑')
    parser.add_argument('--train_folder', required=True, help='訓練資料夾路徑')
    parser.add_argument('--border_size', required=True, help='添加邊框後的圖片大小')
    args = parser.parse_args()

    source_folder = args.source_folder
    target_folder = args.target_folder
    train_folder = args.train_folder
    border_size = int(args.border_size)

    river_folder = os.path.join(target_folder, 'river/test_A')
    road_folder = os.path.join(target_folder, 'road/test_A')

    move_files(source_folder, river_folder, road_folder)
    add_border_to_folder(border_size, river_folder)
    add_border_to_folder(border_size, road_folder)

    ri_train_folder = os.path.join(train_folder, 'river', 'train_A')
    ro_train_folder = os.path.join(train_folder, 'road', 'train_A')

    process_images(river_folder, ri_train_folder, river_folder)
    process_images(road_folder, ro_train_folder, road_folder)

if __name__ == '__main__':
    main()