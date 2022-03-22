Date：[[2022-03-22]]
Link：[[深度学习Pytorch]]/[[linux 系统操作]]

## Python VScode python 关于无法找到文件相对路径的问题 No such file or directory
https://blog.csdn.net/Humphreyr/article/details/121134443
* 改为绝对路径
* 在所配置的 _launch.json_ 文件中的 _configurations_ 列表中加入这一行，~~记得在上一行末尾加上一个逗号~~
	* `“cwd”: “${fileDirname}”`

# VScode Python no module
https://blog.csdn.net/weixin_44560698/article/details/118525324
在lauch.json中加入
`"env": {"PYTHONPATH":"${workspaceRoot}"}`