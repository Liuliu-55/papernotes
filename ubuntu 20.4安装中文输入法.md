Date：[[2022-03-21]]
Link：[[Ubuntu 配置]]

Ubuntu升级到20.4后搜狗输入法不好用，没适配到最新发行版，使用googlepinyin
## 安装fcitx-googlepinyin
Ctrl+Shift+T打开terminal
`sudo apt-get install fcitx-googlepinyin`

## 配置language support
![[Pasted image 20220321220646.png |300]]
语言支持界面中，最下面一行Keyboard input method system，默认是iBus，点击下拉单切换到fcitx（系统初始没有fctix，安装fcitx-googlepinyin的时候会装好fcitx）。然后重启电脑。

## 输入法配置
![[Pasted image 20220321220752.png |300]]
进入configure
点击输入方法设置左下角的+号，进入添加输入方法界面。取消“只显示当前语言”选项的勾选，输入pinyin搜索到系统现有的拼音输入法。选择Google Pinyin并点击OK确认。
![[Pasted image 20220321220905.png]]
关闭设置，谷歌输入法配置完成。可以点击右上角状态栏的键盘图片切换到谷歌输入法，切换输入法的快捷键是ctrl+space，可以在刚关闭的输入方法设置界面里第二项Global Config里修改快捷键。