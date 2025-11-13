快速实现目标检测
=
# 1.目标检测任务
目标检测是计算机视觉中的核心任务，旨在识别和定位图像或视频中的特定物体。它不仅需要判断图像中存在哪些类别的物体（分类任务），还需要通过边界框精确标出每个物体的位置（定位任务）。目标检测技术广泛应用于自动驾驶、视频监控、医疗影像分析和机器人导航等领域，是实现机器“视觉理解”的关键一步。  
YOLO是其中的佼佼者，检测速度和精度都是非常好的，并且实现较为简单。  
# 2.YOLO算法发展历史 (v1-v13)

```mermaid
flowchart TD
    subgraph A[第一代奠基期 2016-2018]
        direction LR
        A1[YOLOv1<br/>2016] --> A2[YOLOv2<br/>2017] --> A3[YOLOv3<br/>2018]
    end

    subgraph B[性能优化期 2020]
        direction LR
        B1[YOLOv4<br/>2020] --> B2[YOLOv5<br/>2020]
    end

    subgraph C[架构革新期 2022-2023]
        direction LR
        C1[YOLOv6<br/>2022] --> C2[YOLOv7<br/>2022] --> C3[YOLOv8<br/>2023]
    end

    subgraph D[前沿探索期 2024]
        direction LR
        D1[YOLOv9<br/>2024] --> D2[YOLOv10<br/>2024] --> D3[YOLOv11<br/>2024] --> D4[YOLOv12<br/>2024] --> D5[YOLOv13<br/>2024]
    end

    A --> B --> C --> D

    %% 第一代奠基期特性
    A1 -->|单阶段回归| A1_1[开创性设计]
    A1 -->|端到端训练| A1_2[速度突破]
    A2 -->|锚框机制| A2_1[定位精度提升]
    A2 -->|多尺度训练| A2_2[Darknet-19]
    A3 -->|特征金字塔| A3_1[多尺度检测]
    A3 -->|残差结构| A3_2[Darknet-53]
    
    %% 性能优化期特性
    B1 -->|BoF/BoS技巧| B1_1[训练优化]
    B1 -->|Mosaic增强| B1_2[数据增强]
    B2 -->|PyTorch实现| B2_1[工程化]
    B2 -->|多模型尺度| B2_2[灵活部署]
    
    %% 架构革新期特性
    C1 -->|RepVGG风格| C1_1[骨干网络革新]
    C2 -->|可训练袋状架构| C2_1[结构创新]
    C3 -->|无锚点设计| C3_1[简化流程]
    C3 -->|多任务支持| C3_2[功能扩展]
    
    %% 前沿探索期特性
    D1 -->|可编程梯度信息| D1_1[梯度优化]
    D2 -->|NMS-free| D2_1[后处理革新]
    D3 -->|小目标检测| D3_1[性能增强]
    D4 -->|注意力机制| D4_1[鲁棒性提升]
    D5 -->|Transformer融合| D5_1[架构探索]
```
# 3.快速实现YOLOv13算法
在基于前面所说的Python相关软件和库安装完成后，我们选择用pip进行安装(以下都可以在PyCharm中运行)  
```
pip install ultralytics
```  
创建一个Python文件,代码如下    
```from ultralytics import YOLO
# Load a pretrained YOLO11n model，如果下载太慢可以点击这个链接下载，并用自己下载的yolo11n.pt所在的文件地址替换，例如“D:\2-Python\1-YOLO\YOLOv11\ultralytics-8.3.2\yolo11n.pt”
model = YOLO("yolo11n.pt")
# Run inference on 'bus.jpg' with arguments,可以将其中图片替换成自己的，需要修改链接
model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5, show=True)
```
  
最后结果：  
<img width="799" height="1042" alt="bus" src="https://github.com/user-attachments/assets/0938207b-c798-4aa7-9a89-389d2b76dea8" />
