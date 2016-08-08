# EDL-python
EDL主要任务是检测句子级别文本中的多种情绪以及各种情绪的强度。  

句子x_i被预测为情绪类别y_j的概率为:

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/1.png)

最优化模型参数θ^∗

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/2.png)

使用L-BFGS的优化算法，采用标签分布学习（Label Distribution Learning, LDL）方法的评估准则，分别为Edulidean、Sϕrensen、Squared X2、Kullback-Leibler (KL)、Intersection、Fidelity。
