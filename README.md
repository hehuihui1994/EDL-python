# EDL-python

------

##简介
> EDL尝试复现论文[Emotion Distribution Learning from Texts](http://www.emnlp2016.net/accepted-papers.html)中的方法，主要任务是
检测文本中的情绪分布以及各个类别情绪强度，使用python编程实现。

##方法

  句子x_i被预测为情绪类别y_j的概率为:

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/1.png)

  最优化模型参数θ^∗

![image](https://github.com/hehuihui1994/EDL-python/blob/master/images/2.png)

  使用L-BFGS的优化算法，采用标签分布学习（Label Distribution Learning, LDL）方法的评估准则，分别为Edulidean、Sϕrensen、Squared X2、Kullback-Leibler (KL)、Intersection、Fidelity。

##文件说明
  
* EDL_hhh.py<br>
  实现标签分布学习（Label Distribution Learning, LDL）方法。
  
* evaluation_for_LDL.py<br>
  计算评估指标Edulidean、Sϕrensen、Squared X2、Kullback-Leibler (KL)、Intersection、Fidelity。
  
* xalglib.py<br>
  alglib库，里面包含很多方法，有多个版本的，其中有LBFGS的方法。
