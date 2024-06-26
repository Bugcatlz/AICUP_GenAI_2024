# AICUP_GenAI_2024
以生成式AI建構無人機於自然環境偵察時所需之導航資訊競賽 I － 影像資料生成競賽

## Competition Background
此競賽為 AI CUP 2024 以生成式 AI 建構無人機於自然環境偵察時所需之導航資訊競賽 I－影像資料生成競賽。作為一項生成式AI比賽，目標是要根據給定的標籤圖，生成與真實空拍圖最相似的圖片，來獲得最小化FID分數。


科技進步雖便利生活，卻也衍生環境永續挑戰。掌握國土環境資訊是實現可持續發展的關鍵。無人機能高效率偵察地形環境，有助獲取環境數據，促進綠能科技及循環經濟發展。但獲取真實影像成本高昂。生成式AI可基於少量數據生成大量逼真影像，本項目將運用此技術模擬生成無人機視野下的道路及河流景象。

## Problem Description
本競賽題目要求參賽者根據給定的黑白標籤圖，透過生成式AI模型生成對應的真實空拍影像。標籤圖中標記了河流或道路的邊界線及中軸線，需要生成能夠模擬無人機視野下的河流或道路實景影像。

-標籤圖

<img src="https://github.com/Bugcatlz/AICUP_GenAI_2024/assets/90192320/a6b4865c-98b1-4f00-8b6c-795283bbcd37" height="200">


-對應的空拍圖片

<img src="https://github.com/Bugcatlz/AICUP_GenAI_2024/assets/90192320/363dd79a-a4a8-4eef-9ef5-e21f07cf0738" height="200">


競賽所使用的訓練資料集是由無人機以程式或人工操控的方式，在台灣各地區如台北、苗栗、台中、台南和屏東等不同縣市的道路及河流進行拍攝獲得。這些影像涵蓋了晴天、陰天及雨後等不同天氣狀況，並使用不同的拍攝角度及影像比例進行拍攝。在標註作業中，使用labelme工具對無人機拍攝影像中的道路及河流以多邊形(polygon)標註其邊界線(border)，並以折線(polyline)標註中軸線(center)。

道路資料集根據天氣、拍攝角度及影像占比等因素共分為12類，河流資料集則分為18類。參賽隊伍需要開發生成式AI模型，能有效從標籤圖生成與真實影像高度相似的空拍影像輸出。
本競賽採用FID(Fréchet Inception Distance)指標對生成影像進行評分。

FID用於計算真實影像和生成影像之特徵分布的距離，分數越低表示生成影像品質越好。相關計算方式如下：


![image](https://github.com/Bugcatlz/AICUP_GenAI_2024/assets/90192320/d5684ee8-0508-4f38-a767-608f7b7519bd)

其中 m 和 m_w 分別表示真實影像分布與生成影像分布的平均值向量；C 和 C_w分別為真實影像分布與生成影像分布的共變異數矩陣(Covariance Matrix)。


最終分數計算的方法為河流影像與道路影像會個別計算一個FID分數，並進行加權評分得到的最終分數FINAL SCORE。

![image](https://github.com/Bugcatlz/AICUP_GenAI_2024/assets/90192320/f389d21e-92eb-408e-9e97-7b96f6f24947)

## Prerequisites
- Linux
- NVIDIA GPU (11G memory or larger) + CUDA cuDNN

## Getting Started

Clone this repo：

```
git clone https://github.com/Bugcatlz/AICUP_GenAI_2024.git
cd AICUP_GenAI_2024
```

Setting environment：
```
conda env create -f environment.yml
conda activate pix2pixHD
```

## Pretrained Model

將解壓縮後的資料夾放置在```./checkpoints```內
| Model Name|Description|Download Link|
|-|-|-|
|river_global_448p|Two-Stage Training|[Link](https://mega.nz/file/eZVnEZ5I#4O3m6hmb9P0BjTXVyDFZE_VRIaet9yeAYDFaP035wPc)|
|river_local_448p|Two-Stage Training|[Link](https://mega.nz/file/rFcBHTiR#Xf6sYvPiylaBdnBMCS94UI97RrsOKNgCoQ-vDPTmZmk)|
|road_global_448p|Two-Stage Training|[Link](https://mega.nz/file/iFk0kIjS#xRNXvvN1tq0QN6YSeZX9JzetlFg7PgDMoDh4c56Fdss)|
|road_local_448p|Two-Stage Training|[LinK](https://mega.nz/file/KUE32I4Z#Qw_DAm-Gn8RSqFLLIBhxT7epDhV2GfDsGPOZOZK0boU)|
|river_local_512p|Unified Training|[Link](https://mega.nz/file/OFsD1Loa#MOtrjvifXbBFgWRwvjuVM62j7oP5Xu69yUABUgSlORE)|
|road_local_512p|Unified Training|[Link](https://mega.nz/file/fNVSFSga#aqH2gCGxkinI9aQQzE_IhPbq6UyYFgdidB-zbkdd3fk)|

註：512p的模型在preprocess時要將```--border_size```設為512，且在inference時將```--loadSize```與```--fineSize```設為512。

## Prepare Training Dataset

首先，從競賽頁面下載訓練資料集並解壓縮。

接著，將訓練資料集分為河流和道路兩個子集，並進一步劃分為訓練集和驗證集。可以使用以下指令來完成這個步驟：
```
python train_preprocess.py --border_size 448 \
                           --source_folder {dataset_path}  \
                           --target_folder {train_split_dataset_path}
```
請將 {dataset_path} 替換為訓練資料集解壓縮後```Traing dataset```資料夾的路徑，{train_split_dataset_path} 替換為劃分後資料集的目標路徑。


## Training (Two-Stage)

我們將河流與道路分開訓練，我們先訓練Global Generator再訓練Local Enhancer。

### River
首先在低分辨率下訓練河流資料集的Global Generator。使用以下指令來完成這個步驟：

```
python train.py --name river_global_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/river \
                --save_epoch 5 \
                --netG global \
                --loadSize 224 \
                --fineSize 224 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 128 \
                --num_D 2 \
                --niter 100 \
                --niter_decay 100
```

訓練完成後，再將訓練後Global Generator的模型作為Local Enhancer的預訓練模型，並在高分辨率下進行訓練。

使用以下指令來完成這個步驟：

```
python train.py --name river_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/river \
                --save_epoch 5 \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 64 \
                --num_D 2 \
                --niter 50 \
                --niter_decay 50 \
                --niter_fix_global 10 \
                --load_pretrain ./checkpoints/river_global_448p
```

### Road
同樣地，首先低分辨率下訓練道路資料集的Global Generator。使用以下指令來完成這個步驟：

```
python train.py --name road_global_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/road \
                --save_epoch 5 \
                --netG global \
                --loadSize 224 \
                --fineSize 224 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 128 \
                --num_D 2 \
                --niter 50 \
                --niter_decay 50
```

訓練完成後，再將訓練後Global Generator的模型作為Local Enhancer的預訓練模型，並在高分辨率下進行訓練。

使用以下指令來完成這個步驟：

```
python train.py --name road_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/road \
                --save_epoch 5 \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 64 \
                --num_D 2 \
                --niter 50 \
                --niter_decay 50 \
                --niter_fix_global 10 \
                --load_pretrain ./checkpoints/road_global_448p
```
若要查看即時的訓練結果，請在 ```./checkpoints/{model_name}/web/index.html``` 中察看。

## Training (Unified Training)

我們將河流與道路分開訓練，但我們直接將Global Generator與Local Enhancer統一訓練。

### River
```
python train.py --name river_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/river \
                --save_epoch 5 \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 64 \
                --num_D 2 \
                --niter 100 \
                --niter_decay 100
```

### Road

```
python train.py --name road_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {train_split_dataset_path}/road \
                --save_epoch 5 \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --no_flip \
                --save_latest_freq 2000 \
                --ngf 64 \
                --num_D 2 \
                --niter 100 \
                --niter_decay 100
```

若要查看即時的訓練結果，請在 ```./checkpoints/{model_name}/web/index.html``` 中察看。

## Inference (public、private data)

首先，從競賽頁面下載測試資料集並解壓縮。

接著，將訓練資料集分為河流和道路兩個子集。可以使用以下指令來完成這個步驟：
```
python test_preprocess.py --border_size 448 \
                          --source_folder {dataset_path} \
                          --target_folder {test_split_dataset_path} \
                          --train_folder {train_split_dataset_path}
```

請將{dataset_path}替換為訓練資料集解壓縮後```label_img```資料夾的路徑。

### River

對於河流資料集可以用以下指令來完成這個步驟：

```
python test.py  --name river_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {test_split_dataset_path}/river \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --ngf 64 \
                --save_output
```

生成的影像會儲存在```./results/river_local_448p/test_latest/synthesized_image```

在執行下列的指令來進行後處理，來滿足競賽要求的圖片格式：

```
python postprocess.py --source_folder ./results/river_local_448p/test_latest/synthesized_image \
                        --target_folder {target_path}
```
請將 {target_folder} 替換為儲存生成結果的資料夾。

### Road

同理，對於道路資料集可以用以下指令來完成這個步驟：

```
python test.py  --name road_local_448p \
                --no_instance \
                --label_nc 0 \
                --dataroot {test_split_dataset_path}/road \
                --netG local \
                --loadSize 448 \
                --fineSize 448 \
                --ngf 64 \
                --save_output
```

生成的影像會儲存在```results/road_local_448p/test_latest/synthesized_image```

在執行下列的指令來進行後處理，來滿足競賽要求的圖片格式：

```
python postprocess.py --source_folder ./results/road_local_448p/test_latest/synthesized_image \
                        --target_folder {target_path}
```
請將 {target_folder} 替換為儲存生成結果的資料夾。

## More Training/Test Details

訓練和測試階段的詳細設置和參數,可以分別參考以下文件:
- 訓練階段設置選項:

  - ```./options/train_options.py```
  - ```./options/base_options.py```

    上述文件中包含了訓練過程中所有可設置的flags選項。

- 測試階段設置選項:

  - ```./options/test_options.py```
  - ```./options/base_options.py```

    這些文件則列出了在測試和推理階段可設置的flags選項。

## Acknowledgments

此專案中的模型修改於 [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)。
