Date：[[2022-03-22]]
Link：

## EESP Block
```python
import torch.nn as nn

class DeepWiseDilationBlock(nn.Module):

"""(depthwise => pointwise)"""

	def __init__(self,in_ch,out_ch,k_size,pad,dil):

		super().__init__()

		self.block = nn.Sequential(
								nn.Conv2d(in_ch, in_ch, kernel_size=k_size, padding=pad, dilation=dil),

								nn.Conv2d(in_ch, out_ch, kernel_size=1,padding=pad, dilation=dil))

	def forward(self, x):

		x=self.block

		return x


class EESPblock(nn.Module):

	def __init__(self,in_ch,out_ch,group=4):

		super().__init__()

		self.ground_conv_1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1,padding=0,dilation=1,groups=group)

		self.deepwise_dilation_conv_block1_3x3 = nn.DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=0,dil=1)

		self.deepwise_dilation_conv_block2_3x3 = nn.DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=0,dil=2)

		self.deepwise_dilation_conv_block3_3x3 = nn.DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=0,dil=3)

		self.deepwise_dilation_conv_block4_3x3 = nn.DeepWiseDilationBlock(out_ch,out_ch,k_size=3,pad=0,dil=4)

  

	def forward(self,x):

		x = self.ground_conv_1x1(x)

		x1 = self.deepwise_dilation_conv_block1_3x3(x)

		x2 = self.deepwise_dilation_conv_block2_3x3(x)

		x3 = self.deepwise_dilation_conv_block3_3x3(x)

		x4 = self.deepwise_dilation_conv_block4_3x3(x)

		x_add1 = torch.add(x1,x2)

		x_add2 = torch.add(x_add1,x3)

		x_add3 = torch.add(x_add2, x4)

		x_cat = torch.cat((x_add1,x_add2,x_add3),1)

		x_gconv = self.ground_conv_1x1(x_cat)

		return x_gconv
```

## SPP Block
