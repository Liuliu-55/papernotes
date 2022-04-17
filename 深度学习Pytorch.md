## Tensorboard👩‍💻
Date : [[2022-03-19]]
Link : [[Deep Learing Model]]
### 简介
tensorboard 是 tensorflow 内置的可视化工具
对 tensorflow 程序输出的日志文件信息进行可视化
***
###  使用逻辑
* 导入 tensorboard 模块
	`form torch.utils.tensorboard import SummaryWriter`
	`SummaryWriter`  的作用是将数据以特定格式存储到指定文件夹中

* 实例化
	- `writer = SummaryWriter(./path/to/log)`
	    传入的参数为指向文件夹路径
	- 针对数值
	    `writer.add_scalar(tag, scalar_value, global_step=None, walltime=None)`
	    `tag` : 可视化的变量名
	    `scalar_value` : 要存的值
	- 针对图像
	    `writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`
	    `tag` : 可视化变量名
	    `img_tensor` : 要存的矩阵
	- 保存日志文件
	    `writer.close()` 
- 可视化
	- 打开存储数据当前文件夹，发现如下文件
	- 打开终端，使用 `tensorboard --logdir=./` 命令，终端操作完后会出现网址，点开浏览器访问地址即可![[Pasted image 20220319140815.png]]
	- 使用完成后，在终端输入`ctrl+C` 即可
	    `writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')`
	    `tag` : 可视化变量名
	    `img_tensor` : 要存的矩阵
## torch.nn🔑
* nn.Maxpool2d
	* stride[a, x, y, z]
		* a：batch滑动步长
		* x：水平滑动步长
		* y：垂直滑动步长
		* z：通道滑动步长
	* dilation
* nn.ConvTranspose2d
* torch.utils.data.Dataloader
* nn.zeropad2d
* torch.nn.function.interpolate
* torch.squeeze

## pytorch导入预训练模型
pytorch导入官方预训练好的Resnet模型
https://blog.csdn.net/weixin_42118374/article/details/103761795
```python
from torchvision import models
import torch.nn as nn


class Module(nn.Module):
	def __init__(self):
		super.__init__()
		self.model = models.resnet50(pretrained=True)
		# 加载resnet50模型
		# pretrained=True会加载训练好的参数
		self.conv_model.load_state_dict(torch.load("路径"))
		# 如果下载了模型，则直接本地加载，pretrained设置为False
	

```

常用的Resnet下载地址：
```ad-note
title:下载地址
 'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
```

