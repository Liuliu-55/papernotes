Date：[[2022-03-22]]
Link：

## EESP Block
```python
import torch.nn as nn

class DeepWiseDilationBlock(nn.Module):

"""(depthwise => pointwise)"""

	def __init__(self,in_ch,out_ch,k_size,pad,dil):

		super().__init__()
		# 两个括号一定要加，不然会报错

		self.block = nn.Sequential(
								nn.Conv2d(in_ch, in_ch, kernel_size=k_size, padding=pad, dilation=dil, group=),

								nn.Conv2d(in_ch, out_ch, kernel_size=1,padding=pad, dilation=dil))

	def forward(self, x):

		x=self.block(x)

		return x


class EESPblock(nn.Module):

	def __init__(self,in_ch,out_ch,group=4):

		super().__init__()

		self.ground_conv_1x1_block1 = nn.Conv2d(in_ch, out_ch, kernel_size=1,padding=0,dilation=1,groups=group)

		self.deepwise_dilation_conv_block1_3x3 = DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=1,dil=1)

		self.deepwise_dilation_conv_block2_3x3 = DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=2,dil=2)

		self.deepwise_dilation_conv_block3_3x3 = DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=3,dil=3)

		self.deepwise_dilation_conv_block4_3x3 = DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=4,dil=4)

		self.ground_conv_1x1_block2 = nn.Conv2d(384, out_ch, kernel_size=1,padding=0,dilation=1,groups=group)

  

	def forward(self,x):

		x = self.ground_conv_1x1_block1(x)

		x1 = self.deepwise_dilation_conv_block1_3x3(x)

		x2 = self.deepwise_dilation_conv_block2_3x3(x)

		x3 = self.deepwise_dilation_conv_block3_3x3(x)

		x4 = self.deepwise_dilation_conv_block4_3x3(x)

		x_add1 = torch.add(x1,x2)

		x_add2 = torch.add(x_add1,x3)

		x_add3 = torch.add(x_add2, x4)

		x_cat = torch.cat((x_add1,x_add2,x_add3),1)

		x_gconv = self.ground_conv_1x1_block2(x_cat)

		return x_gconv```

## SPP Block
```python

import torch.nn as nn
class SPPBlock(nn.Module):

"""input (c,h,w), output an fixed size(c_out,h_out,w_out)"""

def __init__(self, num_levels, pool_type='max_pool'):

super(SPPBlock, self).__init__()

  

self.num_levels = num_levels

self.pool_type = pool_type

  

def forward(self, x):

num, c, h, w = x.size() # num:样本数量 c:通道数 h:高 w:宽

for i in range(self.num_levels):

level = i+1

kernel_size = (math.ceil(h / level), math.ceil(w / level))

stride = (math.ceil(h / level), math.ceil(w / level))

pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

  

# 选择池化方式

if self.pool_type == 'max_pool':

tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

else:

tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

  

# 展开、拼接

if (i == 0):

x_flatten = tensor.view(num, -1)

else:

x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

return x_flatten
```