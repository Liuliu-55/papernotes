### 1.1 服务器后台运行
[Linux nohup 命令 | 菜鸟教程 (runoob.com)](https://www.runoob.com/linux/linux-comm-nohup.html)
使用nohup在终端进行后台运行`nohup [command] &`
```shell
nohup python train.py --cfg config_files/tmvanet.json &
```
输出信息会保存到该运行文件目录下`nohup.out`临时文件中
采用`ps -aux`查看当前运行进程，或者使用`ps -aux |grep "train.py"`查看特定进程
找到PID后可以使用`kill -9 PID`结束进程
![[Selection_008.png]]