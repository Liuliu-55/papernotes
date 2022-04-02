### 1
使用git在笔记本电脑和主机间进行obsidian笔记同步
在主机进行`git push origin master`后笔记本电脑`git pull gitee master`发生unmerge错误
使用`git status`进行故障查询
![[Pasted image 20220402150621.png]]

解决方法：
* 我需要远程分支，本地分支较为老旧，进行丢弃
```shell
git reset --hard FETCH_HEAD
```
FETCH_HEAD 为远程分支最新版本

* 不能丢弃本地更改，先对unmerge文件进行