# -*- coding: utf-8 -*-

'''
test_y[i][j]为实际标签，predict_test_y[i][j]为预测标签
'''

import math

#Euclidean
def Euclidean(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		temp=0
		for j in range(len(test_y[i])):
			temp += ( test_y[i][j] - predict_test_y[i][j])**2
		temp = math.sqrt(temp)
		res += temp
	res = res*1.0/len(test_y)
	return res

#Sϕrensen
def S_rensen(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		A=0
		B=0
		temp=0
		for j in range(len(test_y[i])):
			if ( test_y[i][j] + predict_test_y[i][j] < 0):
				A=A
				B=B
			else:
				A += math.abs(test_y[i][j] - predict_test_y[i][j])
				B += (test_y[i][j] + predict_test_y[i][j])
		temp = A/B
		res += temp
	res = res*1.0/len(test_y)
	return res

#Squared X2
def squaredx2(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		temp=0
		for j in range(len(test_y[i])):
			if (test_y[i][j] + predict_test_y[i][j] <= 0):
				temp=temp
			else:
				temp += ( (test_y[i][j] - predict_test_y[i][j])**2 / ( test_y[i][j] + predict_test_y[i][j]) )
		res += temp
	res = res*1.0/len(test_y)
	return res

#Kullback-Leibler (KL)
def kl(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		temp=0
		for j in range(len(test_y[i])):
			temp += test_y[i][j] * ( math.log(test_y[i][j] / predict_test_y[i][j]) )
		res += temp
	res = res*1.0/len(test_y)
	return res

#Intersection
def intersection(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		temp=0
		for j in range(len(test_y[i])):
			temp += math.min( test_y[i][j] , predict_test_y[i][j] )
		res += temp
	res = res*1.0/len(test_y)
	return res

#Fidelity
def fidelity(predict_test_y,test_y):
	res=0
	for i in range(len(test_y)):
		temp=0
		for j in range(len(test_y[i])):
			temp += math.sqrt(test_y[i][j] * math.abs(predict_test_y[i][j]))
		res += temp
	res = res*1.0/len(test_y)
	return res


if __name__ == '__main__':
	main()