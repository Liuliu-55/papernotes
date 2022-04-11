传感器融合：
[传感器融合：毫米波雷达+摄像头 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/381927166)


涉及坐标变换
* 雷达数据（RA视图）为极坐标
* 光学图像为透视坐标
	* [ika-rwth-aachen/Cam2BEV: TensorFlow Implementation for Computing a Semantically Segmented Bird's Eye View (BEV) Image Given the Images of Multiple Vehicle-Mounted Cameras. (github.com)](https://github.com/ika-rwth-aachen/Cam2BEV)
	* IPM（逆透射变换）：[(19条消息) C++与Python实现逆透视变换IPM（鸟瞰图）_Da_wan的博客-CSDN博客_opencv 透视变换](https://blog.csdn.net/Da_wan/article/details/121481434?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_default&utm_relevant_index=7)

需要将雷达数据与光学数据都转换到BEV（笛卡尔）坐标系下

* 自动驾驶中的BEV表示方法
	* [自动驾驶感知中BEV的景物表示方法（上） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365543182)
	* [自动驾驶感知中BEV的景物表示方法（下） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365561705)