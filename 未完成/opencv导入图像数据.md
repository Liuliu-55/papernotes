Date：[[2022-03-21]]
Link：[[深度学习Pytorch]] / [[Deep Learing Model]]  / [[毕业设计]]

## 图像导入网络
任务：光学图像融入雷达网络中

步骤一：将图像转为Tensor类导入网络中
```python
>>>>>import cv2
>>>>>import numpy as np
>>>>>import torch

>>>>>im = cv2.imread("<地址>",1)
# 1表示图像为RGB图，-1表示图像为灰度图
# opencv 读入RGB图的numpy数据顺序为RBG，需要进行数据转换
>>>>>print(type(im))
<class 'numpy.ndarray'>
>>>>>im.shape
(1028, 1232, 3)

>>>>>im1 = im.transpose(2,0,1)
>>>>>im1.shape
(3, 1028, 1232)

>>>>>im2 = torch.from_numpy(im1)
>>>>>type(im2)
<class 'torch.Tensor'>



```

问题：
```python
>>>>>im = cv2.imread("F:\datasets\datasets_local\Carrada\2019-09-16-12-52-12\camera_images\0000000.jpg",1)
>>>>>>>> type(im)
<class 'NoneType'>
```
windows使用路径时要用双斜杠`\\` 或反斜杠`/`
且路径不能带有中文

## 图像预处理
```python
import cv2
camera_matrix = cv2.imread(os.path.join(self.path_to_frames,
										"camera_images",
										frame_name+".jpg"),0)
# camera_matrix.shape = (1028,1232)
camera_matrix = cv2.resize(camera_matrix,(256,256))
# camera_matrix.shape = (235,256)
```