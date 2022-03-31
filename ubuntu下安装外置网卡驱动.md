## 下载适配与ubuntu20.4的RTL8812AU双频无线网卡驱动
下载地址：[gnab/rtl8812au: Realtek 802.11n WLAN Adapter Linux driver (github.com)](https://github.com/gnab/rtl8812au)
```shell
$ sudo make
$ sudo make install
$ sudo modprobe 8812au
$ sudo lsmod | egrep 8812au
 8812au 856064  0
```

## 使用NetworkManager管理网络
* 查看网络设备列表
```shell
$ sudo nmcli dev 或者
$ nmcli -p device
=====================
  Status of devices
=====================
DEVICE  TYPE      STATE      CONNECTION
----------------------------------------------------------------------
ens33   ethernet  connected  Wired connection 1
lo      loopback  unmanaged  --
```
注意，如果列出的设备状态是 unmanaged 的，说明网络设备不受NetworkManager管理，你需要清空 /etc/network/interfaces下的网络设置,然后重启.
* 开启wifi
```shell
$ sudo nmcli r wifi on
```
* 扫描附WiFi热点
```shell
sudo nmcli dev wifi
```
* 连接到指定WiFi热点
```shell
sudo nmcli dev wifi connect "SSID" password "PASSWORD" ifname wlan0
```
注意：
SSID：实际wifi名字
PASSWORD：实际wifi密码
wlan0：扫描wifi热点时你所要连的wifi名字(eg:ens33)


问题：
电脑跳闸后网卡弹出
进入RTL8812AU驱动文件后重新执行网卡挂载报错
![[Selection_007.png]]
```shell
(base) liuliu@liuliu-Z590-UD:~/rtl8812au$ sudo modprobe 8812au
modprobe: ERROR: could not insert '8812au': Invalid argument
(base) liuliu@liuliu-Z590-UD:~/rtl8812au$ sudo nmcli dev
DEVICE  TYPE      STATE      CONNECTION 
enp4s0  ethernet  connected  有线连接 1 
lo      loopback  unmanaged  --     
```

可以看到无法挂载网卡

解决方法：
在该文件夹内打开终端输入命令
```shell
$ make clean
$ make
$ sudo make install
$ sudo modprobe 8812au
```
再看设备
```shell
(base) liuliu@liuliu-Z590-UD:~/rtl8812au$ sudo nmcli dev
DEVICE           TYPE      STATE      CONNECTION 
enp4s0           ethernet  connected  有线连接 1 
enx502b73c90692  wifi      connected  LLL        
lo               loopback  unmanaged  --   
```
可以看见设备已被挂载，网卡链接wifi成功