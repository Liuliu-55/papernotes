# ubuntu python深度学习环境配置
Date: [[2022-03-12]]
Link: [[linux 系统操作]] / [[Ubuntu 配置]]
（[Ubuntu 20.04 深度学习：从零配置 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/142923008)）
## 1.安装Ubuntu
### 1.1 制作ubuntu u盘启动盘
官方文档制作ubuntu usb启动盘
[Create a bootable USB stick with Rufus on Windows | Ubuntu](https://ubuntu.com/tutorials/create-a-usb-stick-on-windows#1-overview)
***
### 1.2 windows系统内分盘
右键**我的电脑**进入管理，点击磁盘管理
![[Pasted image 20220312130328.png]]
压缩磁盘：压缩一部分ssd作为第二系统主盘；压缩一部分机械硬盘作为外界磁盘。
### 1.3 安装ubuntu系统
[Install Ubuntu desktop | Ubuntu](https://ubuntu.com/tutorials/install-ubuntu-desktop#5-installation-setup)
* 插入u盘，开机后按Bios（F12）键进入选择界面，选择u盘之后u盘内的ubuntu系统
* Installation type中选择something else进行ubuntu磁盘分配
	* ssd中，分配300MB~500MB作为系统引导区
	* 机械硬盘，分配16GB左右作为虚拟内存（logical）
	* 分配剩下ssd内存作为主盘
* 剩下跟着官方教程走
***
### 1.4硬盘管理
* 进入Disks软件（系统自带）
* 选中要管理的硬盘，点击Unmont进行挂载
***
## 2.安装环境
### 2.1 安装显卡驱动，CUDA和cuDNN
关于介绍看这篇blog
[(18条消息) CUDA是什么-CUDA简介_syz201558503103的博客-CSDN博客_cuda是什么意思](https://blog.csdn.net/syz201558503103/article/details/111058193)
* 确定与GPU适配版本
	* GPU-Driver：[Official Drivers | NVIDIA](https://www.nvidia.com/Download/index.aspx)
	* Driver-CUDA：[CUDA Compatibility :: NVIDIA Data Center GPU Driver Documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#overview)
	* cuDNN：[NVIDIA Developer Program 需要会员资格 | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download)
* `sudo apt-get install build-essential` 进行底层开发环境安装
* 搜索**NVIDA cuda installation guide for linux** 进入官网，选择对应驱动版本，根据指导进行安装[Installation Guide Linux :: CUDA Toolkit Documentation (nvidia.com)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
	* 禁用集成显卡
	* `sudo sh XXX.run`运行文件
	* `sudo dpkg -i XXX.deb`运行文件
	* 重启电脑
	* `nvidia -smi` 查看GPU情况
***
### 2.2 安装pytorch环境
* 安装anaconda
* 找到GPU适配pytorch/Tenserflow版本
	* Tensorflow-CUDA：[www.tensorflow.org](https://www.tensorflow.org/install/source)
	* Pytorch-CUDA：[Start Locally | PyTorch](https://pytorch.org/get-started/locally/)
* 配置pytorch环境
	按照官网指示进行安装[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)
* 快捷键
	* `TREL+SHIFT+L` 标记选中后所有变量
	* `TREL+D` 跳转选中变量
	* `F12` 查看选中变量来源
***
### 2.3 配置vscode编译器
* 官网下载vscode for linux文件
* `sudo dpkg -i XXX.deb`运行文件进行安装
* 终端输入**code**进入vscode
* 安装python，coderunner等插件
***
### 2.4 安装git环境
* `sudo apt install git`配置git（代码版本管理软件） 
* 生成ssh公钥，连接git以及gitee
***
## 3.远程操纵设置
### 3.1 安装TeamViewer
远程远程操作电脑软件
[远程神器！教你用TeamViewer实现远程开机&无人值守 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/66350624)



***
### 3.2 连接服务器
软件：MobaXterm

