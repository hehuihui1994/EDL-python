# -*- coding: utf-8 -*-

import math
import xalglib
import evaluation_for_LDL

#全局变量
xs=[]
ys=[]
test_x=[]
test_y=[]
xi_1=0
xi_2=0
n=0

#Hypothesis : h(x) = theta[0]+theta[1]*x_1+...
#change sentence to x[]_100  x_i
def H(theta,x):
	res=0
	for i in range(len(x)):
		res+=theta[i]*x[i]
	return math.exp(res)

#emotion distribution of x_i
def p(thetas,x):
	p_x=[]
	p_x_temp=[]
	sum1=0
	for theta in thetas:
		temp=H(theta,x)
		p_x_temp.append(temp)
		sum1+=temp
	for item in p_x_temp:
		p_x.append(item*1.0/sum1)
	return p_x

#z_i  x_i
def z(thetas,x):
	res=0
	for theta in thetas:
		temp=H(theta,x)
		res+=temp
	return res

#两个vec相减后的欧几里得的平方
def Euclid(theta1,theta2):
	res=0
	for i in range(len(theta1)):
		res+=(theta1[i] - theta2[i])**2
	return res


#cost function  T(thetas)
#xs_i  [..,..,..]
#ys_i  [..,..,..]
def T(thetas):
	#i:sentence_i   x_i
	#j: emotion_j
	first=0
	sencond=0
	third=0
	for i in range(xs):
		for j in range(ys[i]):
			first+=(1-(4 * z(thetas, xs[i]) * ys[i][j] * H(thetas[j], xs[i]))*1.0 / 
				    (z(thetas, xs[i]) * ys[i][j] + H(thetas[j], xs[i]))**2 )
	first=2*first
	for k in range(len(thetas)):
		for r in range(len(thetas[k])):
			sencond+=abs(thetas[k][r])
	sencond=(xi_1 / n )* sencond
	#模拟情绪分布中情绪之间的关系
	# Plutchik’s wheel of emotions
	#8种情绪
	w=[[0 for j in range(len(ys[0]))] for i in range(len(ys[0]))]
	temp_w=[1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5]
	index=0
	for i in range(len(w)):
		for j in range(len(w[i])):
			w[i][j]=temp_w[index]
			index+=1
			index=index%8
		index=(index+8-1)%8
	for j in range(len(thetas)):
		for k in range(len(thetas)):
			third += w[j][k] * Euclid(thetas[j], thetas[k])
	third = (xi_2 / n) * third
	res = first + sencond + third
	return res

def sgn(x):
	res=0
	if x > 0:
		res=1
	elif x < 0:
		res=-1
	else:
		res=0
	return res

def get_grad(thetas):
	#计算一阶导
	#thetas以矩阵的形式
	grad = [ [0 for col in range(len(xs[0]))] for row in range(8) ]
	for j in range(len(thetas)):
		for r in range(len(thetas[j])):
			first=0
			sencond=0
			third=0
			for i in range(len(xs)):
				p_x = p(thetas,xs[i])
				p_ij = p_x[j]
				first += ( ys[i][j] * p_ij * (1-p_ij ) * (ys[i][j] - p_ij ) * xs[i][r] ) / ( ys[i][j] + p_ij )**3
			first = first*(-8)
			sencond = (xi_1 / n) * sgn(thetas[j][r])
			#8种情绪
			w=[[0 for j in range(len(ys[0]))] for i in range(len(ys[0]))]
			temp_w=[1, 0.5, 0, -0.5, -1, -0.5, 0, 0.5]
			index=0
			for i in range(len(w)):
				for j in range(len(w[i])):
					w[i][j]=temp_w[index]
					index+=1
					index=index%8
				index=(index+8-1)%8
			for k in range(len(thetas)):
				third += w[j][k] * 2 * (thetas[j][r] - thetas[k][r])
			third = (xi_2/n)*third
			grad[j][r] = first + sencond + third
	return grad


#矩阵转换成vec
def matrix_to_vec(x_before):
	x=[]
	for i in range(len(x_before)):
		for j in range(len(x_before[i])):
			x.append(x_before[i][j])
	return x

#vec转换为矩阵
def vec_to_matrix(x,col):
	x_after=[]
	x_temp=[]
	for i in range(len(x)):
		x_temp.append(x[i])
		if i%col==1 and len(x_temp)!=0:
			x_after.append(x_temp)
			x_temp=[]
	return x_after


def function1_grad(thetas_v, grad, param):
	#传入的thetas是vec
	thetas_m = vec_to_matrix(thetas_v,len(xs[0]))
	func = T(thetas_m,xs,ys,xi_1n,xi_2n)
	grad_before = get_grad(thetas_m)
	grad1 = matrix_to_vec(grad_before)
	#vec形式
	for i in range(len(grad1)):
	    grad[i] = grad1[i]
	return func

#L-BFGS解出最优解
def get_thetas_by_Lbfgs():
	#8种情绪  thetas初始值
	thetas_m = [ [0 for col in range(len(xs[0]))] for row in range(8) ]
	thetas_v = matrix_to_vec(thetas_m)
	epsg = 0.0000000001
	epsf = 0
	epsx = 0
	maxits = 0
	state = xalglib.minlbfgscreate(7, thetas_v)
	xalglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits)
	xalglib.minlbfgsoptimize_g(state, function1_grad)
	thetas_v, rep = xalglib.minlbfgsresults(state)
	print(rep.terminationtype) 
	# print(thetas)
	thetas_m = vec_to_matrix(thetas_v) 
	print thetas_m
	return thetas_m

#测试集合预测
def get_predict_test(thetas):
	#test_x
	predict_test_y=[]
	for x in test_x:
		p_x = p(thetas,x)
		predict_test_y.append(p_x)
	return predict_test_y


if __name__ == '__main__':
	print("训练集数据读取")
	print("测试集数据读取")
	print("求解thetas---")
	thetas=get_thetas_by_Lbfgs()
	print("预测测试集-----")
	predict_test_y=get_predict_test(thetas)
	print("评价指标——-----")
	print("Euclidean : %r"%(evaluation_for_LDL.Euclidean(predict_test_y,test_y)))
	print("Sϕrensen : %r"%(evaluation_for_LDL.S_rensen(predict_test_y,test_y)))
	print("Squared X2 : %r"%(evaluation_for_LDL.squaredx2(predict_test_y,test_y)))
	print("Kullback-Leibler (KL) : %r"%(evaluation_for_LDL.kl(predict_test_y,test_y)))
	print("intersection : %r"%(evaluation_for_LDL.intersection(predict_test_y,test_y)))
	print("fidelity : %r"%(evaluation_for_LDL.fidelity(predict_test_y,test_y)))