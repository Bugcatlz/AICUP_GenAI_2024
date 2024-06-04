import os
import random
import shutil
import cv2
import argparse

def split_data(source_folder, ri_folder, ro_folder):
    """將源資料夾中的檔案複製到 RI 和 RO 資料夾中"""
    files = os.listdir(source_folder)
    for file in files:
        if 'RI' in file and 'RO' not in file and file.endswith(('.jpg', '.png')):
            shutil.copy(os.path.join(source_folder, file), os.path.join(ri_folder, file))
        elif 'RO' in file and file.endswith(('.jpg', '.png')):
            shutil.copy(os.path.join(source_folder, file), os.path.join(ro_folder, file))

def move_data(img_folder, train_folder_img, label_folder, train_folder_label, train_files):
    """將資料移動到訓練和測試資料夾中"""
    for img_file in train_files:
        src_img_path = os.path.join(img_folder, img_file)
        dest_img_path = os.path.join(train_folder_img, img_file)
        src_label_path = os.path.join(label_folder, img_file.replace('.jpg', '.png'))
        dest_label_path = os.path.join(train_folder_label, img_file.replace('.jpg', '.png'))
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dest_img_path)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)

def modified_data_type(train_folder_img, train_folder_label):
    """將訓練和測試資料夾中的標籤檔案名稱從.png修改為.jpg"""
    for folder_path in [train_folder_img, train_folder_label]:
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                old_file_path = os.path.join(folder_path, filename)
                new_file_path = os.path.join(folder_path, filename.replace('.png', '.jpg'))
                os.rename(old_file_path, new_file_path)

def add_border_to_folder(border_size, folder, color):
    """為資料夾中的圖像添加邊框"""
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            top_border = (border_size - image.shape[0]) // 2
            bottom_border = border_size - image.shape[0] - top_border
            left_border = (border_size - image.shape[1]) // 2
            right_border = border_size - image.shape[1] - left_border
            image_with_border = cv2.copyMakeBorder(image, top_border, bottom_border, left_border, right_border,
                                                   cv2.BORDER_CONSTANT, value=color)
            output_path = os.path.join(folder, filename)
            if os.path.exists(output_path):
                os.remove(output_path)
            cv2.imwrite(output_path, image_with_border)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', required=True, help='源資料夾路徑')
    parser.add_argument('--target_folder', required=True, help='目標資料夾路徑')
    parser.add_argument('--border_size', required=True, help='添加邊框後的圖片大小')
    args = parser.parse_args()

    source_folder = args.source_folder
    target_folder = args.target_folder
    border_size = args.border_size

    source_img_folder = os.path.join(source_folder, 'img')
    source_label_folder = os.path.join(source_folder, 'label_img')

    # 建立必要的資料夾
    ri_img_folder = os.path.join(target_folder, 'river')
    ro_img_folder = os.path.join(target_folder, 'road')
    os.makedirs(ri_img_folder, exist_ok=True)
    os.makedirs(ro_img_folder, exist_ok=True)

    # 設置訓練和測試資料夾路徑
    ri_train_folder_label = os.path.join(ri_img_folder, 'train_A')
    ri_test_folder_label = os.path.join(ri_img_folder, 'test_A')
    ri_train_folder_img = os.path.join(ri_img_folder, 'train_B')
    ri_test_folder_img = os.path.join(ri_img_folder, 'test_B')
    ro_train_folder_label = os.path.join(ro_img_folder, 'train_A')
    ro_test_folder_label = os.path.join(ro_img_folder, 'test_A')
    ro_train_folder_img = os.path.join(ro_img_folder, 'train_B')
    ro_test_folder_img = os.path.join(ro_img_folder, 'test_B')
    os.makedirs(ri_train_folder_label, exist_ok=True)
    os.makedirs(ri_test_folder_label, exist_ok=True)
    os.makedirs(ri_train_folder_img, exist_ok=True)
    os.makedirs(ri_test_folder_img, exist_ok=True)
    os.makedirs(ro_train_folder_label, exist_ok=True)
    os.makedirs(ro_test_folder_label, exist_ok=True)
    os.makedirs(ro_train_folder_img, exist_ok=True)
    os.makedirs(ro_test_folder_img, exist_ok=True)

    # 分類資料
    split_data(source_img_folder, ri_train_folder_img, ro_train_folder_img)
    split_data(source_label_folder, ri_train_folder_label, ro_train_folder_label)

    # 分割訓練和測試數據
    ri_img_files = os.listdir(ri_train_folder_img)
    ro_img_files = os.listdir(ro_train_folder_img)
    random.seed(42)
    random.shuffle(ro_img_files)
    random.shuffle(ri_img_files)
    ri_total_files = len(ri_img_files)
    ro_total_files = len(ro_img_files)
    train_ratio = 0.9
    test_ratio = 0.1
    # 計算各個資料集的數量
    ri_train_count = int(ri_total_files * train_ratio)
    ri_test_count = ri_total_files - ri_train_count
    ro_train_count = int(ro_total_files * train_ratio)
    ro_test_count = ro_total_files - ro_train_count

    # 分配訓練和測試數據
    ri_train_files = ri_img_files[:ri_train_count]
    ri_test_files = ri_img_files[ri_train_count:]
    ro_train_files = ro_img_files[:ro_train_count]
    ro_test_files = ro_img_files[ro_train_count:]

    # 移動訓練和測試數據
    move_data(ri_train_folder_img, ri_train_folder_img, ri_train_folder_label, ri_train_folder_label, ri_train_files)
    move_data(ri_train_folder_img, ri_test_folder_img, ri_train_folder_label, ri_test_folder_label, ri_test_files)
    move_data(ro_train_folder_img, ro_train_folder_img, ro_train_folder_label, ro_train_folder_label, ro_train_files)
    move_data(ro_train_folder_img, ro_test_folder_img, ro_train_folder_label, ro_test_folder_label, ro_test_files)

    # 修改檔案類型
    modified_data_type(ri_train_folder_img, ri_train_folder_label)
    modified_data_type(ri_test_folder_img, ri_test_folder_label)
    modified_data_type(ro_train_folder_img, ro_train_folder_label)
    modified_data_type(ro_test_folder_img, ro_test_folder_label)

    # 添加邊框
    add_border_to_folder(border_size, ri_train_folder_img, (0, 0, 0))  # black border
    add_border_to_folder(border_size, ri_train_folder_label, (0, 0, 255))  # red border
    add_border_to_folder(border_size, ri_test_folder_img, (0, 0, 0))  # black border
    add_border_to_folder(border_size, ri_test_folder_label, (0, 0, 255))  # red border
    add_border_to_folder(border_size, ro_train_folder_img, (0, 0, 0))  # black border
    add_border_to_folder(border_size, ro_train_folder_label, (0, 0, 255))  # red border
    add_border_to_folder(border_size, ro_test_folder_img, (0, 0, 0))  # black border
    add_border_to_folder(border_size, ro_test_folder_label, (0, 0, 255))  # red border

if __name__ == '__main__':
    main()
