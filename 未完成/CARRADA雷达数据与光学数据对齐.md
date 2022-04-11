毫米波雷达公开数据库：
[毫米波雷达：公开数据库 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/372804658)

传感器融合：
[传感器融合：毫米波雷达+摄像头 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/381927166)
（很重要）


涉及坐标变换
需要将雷达数据与光学数据都转换到BEV（笛卡尔）坐标系下
* 雷达数据（RA视图）为极坐标
	* [(19条消息) Python-openCV极坐标变换（坐标变换）_Tina-的博客-CSDN博客_python 极坐标转换](https://blog.csdn.net/clm1206/article/details/79786107?spm=1001.2101.3001.6650.10&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-10.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-10.pc_relevant_antiscanv2&utm_relevant_index=17)
	* [(19条消息) 【OpenCV 例程200篇】36. 直角坐标与极坐标的转换_小白YouCans的博客-CSDN博客_opencv 极坐标转换](https://blog.csdn.net/youcans/article/details/121416883)
* 光学图像为透视坐标
	* [ika-rwth-aachen/Cam2BEV: TensorFlow Implementation for Computing a Semantically Segmented Bird's Eye View (BEV) Image Given the Images of Multiple Vehicle-Mounted Cameras. (github.com)](https://github.com/ika-rwth-aachen/Cam2BEV)
	* IPM（逆透射变换）：[(19条消息) C++与Python实现逆透视变换IPM（鸟瞰图）_Da_wan的博客-CSDN博客_opencv 透视变换](https://blog.csdn.net/Da_wan/article/details/121481434?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-4.pc_relevant_default&utm_relevant_index=7)
	* [透视变换之变换为鸟瞰图 - 简书 (jianshu.com)](https://www.jianshu.com/p/b49f9dbb26ea)



* 自动驾驶中的BEV表示方法
	* [自动驾驶感知中BEV的景物表示方法（上） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365543182)
	* [自动驾驶感知中BEV的景物表示方法（下） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365561705)