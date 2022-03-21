Date：[[2022-03-21]]
Link：[[Ubuntu 配置]] / [[ubuntu安装obsidian软件]] / [[linux 系统操作]]

前提：
* windows/ubuntu 平台安装git
* windows平台与ubuntu平台都已经使用ssh密钥连接gitee/github


同步：
* 安装## [obsidian-proxy-github](https://gitee.com/juqkai/obsidian-proxy-github "obsidian-proxy-github") 库包解决obsidian第三方插件无法打开问题
	* 下载后直接移动到`.obsidian/plugins` 下
	* 重启obsidian
* 安装obsidian-git插件
* 在Gitee/GitHub上远程同步
```shell
$ git remote add gitee git@github.com:Liuliu-55/papernotes
error: failed to push some refs to 'https://gitee.com/Liuliu-55/papernotes'
(出错：远程库与本地库不同步)
$ git pull --rebase gitee master
fatal: Updating an unborn branch with changes added to the index.
(将远程仓库中的更新合并到本地仓库，–[rebase]的作用是取消掉本地仓库中刚刚的commit)
(出错：没有commit)
$ git commit -m "..."
$ git pull --rebase gitee master
$ git push gitee master
//
```

问题：使用git上传项目时弹出GitHub登录框
解决：将上传代码方式从http改为ssh
```shell
$ git remote rm gitee@https//gitee.com//Liuliu-55//papernotes
$ git remote add gitee git@github.com:Liuliu-55/papernotes
$ git remote -v
```

git clone时太慢可以使用镜像网站路径
[Mirror List (library.ac.cn)](https://www.library.ac.cn/)
