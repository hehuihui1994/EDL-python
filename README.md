# EDL-python

------

##简介
> EDL尝试复现论文[Emotion Distribution Learning from Texts](http://www.emnlp2016.net/accepted-papers.html)中的方法，使用python编程实现。

##

句子x_i被预测为情绪类别y_j的概率为:

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/1.png)

最优化模型参数θ^∗

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/2.png)

使用L-BFGS的优化算法，采用标签分布学习（Label Distribution Learning, LDL）方法的评估准则，分别为Edulidean、Sϕrensen、Squared X2、Kullback-Leibler (KL)、Intersection、Fidelity。

##文件说明
  
* EDL_hhh.py
  A script used to get dictionary information. 
  
* evaluation_for_LDL.py<br>
  CWSperceptron with 10+punc features.
  
* xalglib.py<br>
  CWSperceptron with 10+punc+dict+type features.
