# crazy实物实验
## 实验方案：
1. 实现无风场干扰的轨迹跟踪
    1. 无人机控制接口
    2. 状态获取以及精度
    3. 代码对接
2. 实现有风场干扰（多方向风场）的轨迹跟踪
    1. GP的C++化实现/并行化
    2. GP预训练（实际采数据训练）
3. 实现有障碍物的避障（暂缓）
    1. IRIS算法的C++代码调试
## 2020/5/19
1. python代码GP的多进程并行计算：无法实现GP类实例共享。
2. python代码GP的多线程并行计算：已实现，但是计算时间并没有得到优化（只是把更新的一大段时间平摊到各个timestep中，且python多线程不是真正意义的多线程），进一步考虑C++化或者其他库
3. GP 实现从多输入单输出变成多输入多输出，只有一个高斯模型

## Add branch zhenglei
