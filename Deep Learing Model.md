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