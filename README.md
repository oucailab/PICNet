# 🚀 Prototype-Based Information Compensation Network for Multi-Source Remote Sensing Data Classification, IEEE TGRS 2025  

[![IEEE TGRS](https://img.shields.io/badge/IEEE-TGRS-blue)](https://ieeexplore.ieee.org/document/11002550/)  [![arXiv](https://img.shields.io/badge/arXiv-2505.04003-b31b1b)](https://arxiv.org/abs/2505.04003) 



## 📌 **Introduction**

This repository contains the official implementation of our paper:  
📄 *Prototype-Based Information Compensation Network for Multi-Source Remote Sensing Data Classification, IEEE TGRS 2025*   

Prototype-based Information Compensation Network ( **PICNet** ) is designed for land cover classification based on HSI and SAR/LiDAR data. Specifically, we first design a **frequency interaction module** to enhance the inter-frequency coupling in multi-source feature extraction. The multi-source features are first decoupled into high- and low-frequency components. Then, these features are recoupled to achieve efficient inter-frequency communication. Afterward, we design a **prototype-based information compensation module** to model the global multi-source complementary information. Two sets of learnable modality prototypes are introduced to represent the global modality information of multi-source data.


---

### 🔍 **Key Features**

🔍 **Key Features:**  
✅ Cross-Modal Remote Sensing Data Joint Classification  
✅ Frequency interaction module  
✅ Prototype-based information compensation module  

---

## 📂 **Dataset**  

The dataset used in our experiments can be accessed from the following link:  
📥 **[Download Dataset Berlin and Augsburg](https://github.com/zhu-xlab/augsburg_Multimodal_Data_Set_MDaS)** 

---

## 🏋️‍♂️ **Usage: Training PICNet**

To train **PICNet** , use the following command:

```bash
python task.py
```

---

## 📬 **Contact**

If you have any questions, feel free to contact us via Email:  
📧 Feng Gao: gaofeng@ouc.edu.cn  
📧 Sheng Liu: jinxuepeng@stu.ouc.edu.cn   
📧 Chuanzheng Gong:  gongchuanzheng@stu.ouc.edu.com

We hope **PICNet** helps your research! ⭐ If you find our work useful, please cite:

```
@ARTICLE{picnet2025,
  author={Gao, Feng and Liu, Sheng and Gong, Chuanzheng and Zhou, Xiaowei and Wang, Jiayi and Dong, Junyu and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Prototype-Based Information Compensation Network for Multi-Source Remote Sensing Data Classification}, 
  year={2025},
  volume={63},
  pages={1-15},
```
