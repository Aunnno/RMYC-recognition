### RMYC-recognizaton
---
#### 介绍
该仓库为RMYC的视觉识别提供解决方案  
* `capture`文件夹为训练用数据的采集方案(里面参数都是依照树莓派的usb摄像头设定的)
* `yolo_recognition`文件夹为图像的识别方案  
* `yolo_train`文件夹为模型的训练方案
#### 需求
* Python>=3.10  
库依赖见`requirements.txt`  
#### 环境配置
* `ultralytics`包含机器学习相关框架  
* `opencv-python`用于视觉处理  
**注意!!!**  
`ultralytics`库中包含依赖`Pytorch`(由`torch`,`torchvision`,`torchaudio`组成)，此为跑深度学习的关键，必须根据你所使用的设备来安装相关版本，比如我的树莓派是aarch64架构，就得安装适用于aarch的pytorch，我的电脑GPU是3070laptop，是n卡且安装了CUDA所以必须安装适用于CUDA的pytorch  
建议安装完Pytorch后再安装ultralytics，不然安装ultralytics的时候会自动安装Pytorch，但是有概率不适用于你的设备
**请自行前往Pytorch官网上查询对应设备的安装命令!!!**
安装好Pytorch后，就要安装ultralytics和opencv了  
```shell
pip install ultralytics
pip install opencv-python
```
如果遇见下载慢/下载失败，建议换源安装  
#### 文件目录结构搭建
* `captured`  
不作处理
* `yolo_recognition`  
需在目录下创建`model`和`photos`文件夹存放模型和待检测图片  
* `yolo_train`  
训练集目录遵循`COCO`格式
```
./
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```
或
```
./
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```
#### 如何训练模型
先配置data.yaml文件(相关路径遵循你所搭建的文件目录结构)  
```yaml
path : "../yolo_train" #根据你自己的根目录
train : "images/train" #根据你自己的结构
val : "images/val"
nc : 1 #根据你自己定义的标签数量
names : ['name'] #根据你自己定义的标签名称
```
配置好data.yaml命令后，运行`train.py`
```shell
python train.py
```
训练参数设置调整请自行查阅yolo官方文档中的train部分