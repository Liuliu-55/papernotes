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
								nn.Conv2d(in_ch, in_ch, kernel_size=k_size, padding=pad, dilation=dil, groups=in_ch),

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

问题：出现Runtime Error
解决：加上`torch.cuda.current_device()`语句

## ConvLSTM
链接：https://github.com/Hzzone/Precipitation-Nowcasting
[GitHub - happyjin/ConvGRU-pytorch: Convolutional GRU](https://github.com/happyjin/ConvGRU-pytorch)
原理：[(19条消息) 【论文翻译】Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting_页页读的博客-CSDN博客](https://blog.csdn.net/u014386899/article/details/100560734)
```python
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        # N=(W?F+2P)/S+1
        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=False)

        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=False)

        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=False)

        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels,
                             self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
        self.Wcf = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
        self.Wco = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
        return torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(), \
               torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda()


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 step=2,
                 effective_step=[1],
                 bias=True):
        """
        :param input_channels: 输入通道数
        :param hidden_channels: 隐藏通道数, 是个列表, 可以表示这个ConvLSTM内部每一层结构
        :param kernel_size: 卷积实现对应的核尺寸
        :param step: 该ConvLSTM自身总的循环次数
        :param effective_step: 输出中将要使用的步数(不一定全用)
        :param bias: 各个门的偏置项
        """
        super(ConvLSTM, self).__init__()

        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i],
                                self.hidden_channels[i],
                                self.kernel_size,
                                self.bias)
            # 设定 self.cell{i} = cell 很好的方法, 值得借鉴, 后期添加属性
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            """
            每个时间步里都要进行对原始输入`input`的多个ConvLSTMCell的的级联处理.
            而第一个时间步里, 设定各个ConvLSTMCell所有的初始h与c都是0.
            各个ConvLSTMCell的输出h和c都是下一个时间步下对应的ConvLSTMCell的输入用的h和c, 
            各个ConvLSTMCell的输入都是同一时间步下上一个ConvLSTMCell的输出的h(作为input项)
            和自身对应的h和c.
            """
            x = input

            # 对每种隐藏状态尺寸来进行叠加
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = f'cell{i}'

                # 初始化各个ConvLSTM的门里的Peehole权重为0
                if step == 0:
                    bsize, _, height, width = x.size()

                    # getattr获得了对应的self.cell{i}的值, 也就是对应的层
                    (h, c) = getattr(self, name).init_hidden(
                        batch_size=bsize,
                        hidden=self.hidden_channels[i],
                        shape=(height, width)
                    )
                    # 第一步里的h和c都是0
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                # update new h&c
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)

```


## ConvGRU
链接：https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
原理：[ConvGRU(ConvLSTM)神经网络的介绍 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/398544620)
```python
import os

import torch

from torch import nn

from torch.autograd import Variable

class ConvGRUCell(nn.Module):

def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):

"""

Initialize the ConvLSTM cell

:param input_size: (int, int)

Height and width of input tensor as (height, width).

:param input_dim: int

Number of channels of input tensor.

:param hidden_dim: int

Number of channels of hidden state.

:param kernel_size: (int, int)

Size of the convolutional kernel.

:param bias: bool

Whether or not to add the bias.

:param dtype: torch.cuda.FloatTensor or torch.FloatTensor

Whether or not to use cuda.

"""

super(ConvGRUCell, self).__init__()

self.height, self.width = input_size

self.padding = kernel_size[0] // 2, kernel_size[1] // 2

self.hidden_dim = hidden_dim

self.bias = bias

self.dtype = dtype

self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,

out_channels=2*self.hidden_dim, # for update_gate,reset_gate respectively

kernel_size=kernel_size,

padding=self.padding,

bias=self.bias)

self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,

out_channels=self.hidden_dim, # for candidate neural memory

kernel_size=kernel_size,

padding=self.padding,

bias=self.bias)

def init_hidden(self, batch_size):

return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

def forward(self, input_tensor, h_cur):

"""

:param self:

:param input_tensor: (b, c, h, w)

input is actually the target_model

:param h_cur: (b, c_hidden, h, w)

current hidden and cell states respectively

:return: h_next,

next hidden state

"""

combined = torch.cat([input_tensor, h_cur], dim=1)

combined_conv = self.conv_gates(combined)

gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)

reset_gate = torch.sigmoid(gamma)

update_gate = torch.sigmoid(beta)

combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)

cc_cnm = self.conv_can(combined)

cnm = torch.tanh(cc_cnm)

h_next = (1 - update_gate) * h_cur + update_gate * cnm

return h_next

class ConvGRU(nn.Module):

def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,

dtype, batch_first=False, bias=True, return_all_layers=False):

"""

:param input_size: (int, int)

Height and width of input tensor as (height, width).

:param input_dim: int e.g. 256

Number of channels of input tensor.

:param hidden_dim: int e.g. 1024

Number of channels of hidden state.

:param kernel_size: (int, int)

Size of the convolutional kernel.

:param num_layers: int

Number of ConvLSTM layers

:param dtype: torch.cuda.FloatTensor or torch.FloatTensor

Whether or not to use cuda.

:param alexnet_path: str

pretrained alexnet parameters

:param batch_first: bool

if the first position of array is batch or not

:param bias: bool

Whether or not to add the bias.

:param return_all_layers: bool

if return hidden and cell states for all layers

"""

super(ConvGRU, self).__init__()

# Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers

kernel_size = self._extend_for_multilayer(kernel_size, num_layers)

hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

if not len(kernel_size) == len(hidden_dim) == num_layers:

raise ValueError('Inconsistent list length.')

self.height, self.width = input_size

self.input_dim = input_dim

self.hidden_dim = hidden_dim

self.kernel_size = kernel_size

self.dtype = dtype

self.num_layers = num_layers

self.batch_first = batch_first

self.bias = bias

self.return_all_layers = return_all_layers

cell_list = []

for i in range(0, self.num_layers):

cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]

cell_list.append(ConvGRUCell(input_size=(self.height, self.width),

input_dim=cur_input_dim,

hidden_dim=self.hidden_dim[i],

kernel_size=self.kernel_size[i],

bias=self.bias,

dtype=self.dtype))

# convert python list to pytorch module

self.cell_list = nn.ModuleList(cell_list)

def forward(self, input_tensor, hidden_state=None):

"""

:param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not

extracted features from alexnet

:param hidden_state:

:return: layer_output_list, last_state_list

"""

if not self.batch_first:

# (t, b, c, h, w) -> (b, t, c, h, w)

input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

# Implement stateful ConvLSTM

if hidden_state is not None:

raise NotImplementedError()

else:

hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

layer_output_list = []

last_state_list = []

seq_len = input_tensor.size(1)

cur_layer_input = input_tensor

for layer_idx in range(self.num_layers):

h = hidden_state[layer_idx]

output_inner = []

for t in range(seq_len):

# input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function

h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], # (b,t,c,h,w)

h_cur=h)

output_inner.append(h)

layer_output = torch.stack(output_inner, dim=1)

cur_layer_input = layer_output

layer_output_list.append(layer_output)

last_state_list.append([h])

if not self.return_all_layers:

layer_output_list = layer_output_list[-1:]

last_state_list = last_state_list[-1:]

return layer_output_list, last_state_list

def _init_hidden(self, batch_size):

init_states = []

for i in range(self.num_layers):

init_states.append(self.cell_list[i].init_hidden(batch_size))

return init_states

@staticmethod

def _check_kernel_size_consistency(kernel_size):

if not (isinstance(kernel_size, tuple) or

(isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):

raise ValueError('`kernel_size` must be tuple or list of tuples')

@staticmethod

def _extend_for_multilayer(param, num_layers):

if not isinstance(param, list):

param = [param] * num_layers

return param

if __name__ == '__main__':

# set CUDA device

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# detect if CUDA is available or not

use_gpu = torch.cuda.is_available()

if use_gpu:

dtype = torch.cuda.FloatTensor # computation in GPU

else:

dtype = torch.FloatTensor

height = width = 6

channels = 256

hidden_dim = [32, 64]

kernel_size = (3,3) # kernel size for two stacked hidden layer

num_layers = 2 # number of stacked hidden layer

model = ConvGRU(input_size=(height, width),

input_dim=channels,

hidden_dim=hidden_dim,

kernel_size=kernel_size,

num_layers=num_layers,

dtype=dtype,

batch_first=True,

bias = True,

return_all_layers = False)

batch_size = 1

time_steps = 1

input_tensor = torch.rand(batch_size, time_steps, channels, height, width) # (b,t,c,h,w)

layer_output_list, last_state_list = model(input_tensor)
```

