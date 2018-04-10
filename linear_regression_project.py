# !/usr/bin/python
# -*- coding: utf-8 -*-

# 任意选一个你喜欢的整数，这能帮你得到稳定的结果
seed = 1

# 1.1 创建一个 4*4 单位矩阵
I = [[1,0,0,0],
	 [0,1,0,0],
	 [0,0,1,0],
	 [0,0,0,1]
]

# 1.2 返回矩阵的行数和列数
def shape(M):
    return len(M),len(M[0])


# 1.3 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for i in range(len(M)):
        for j in range(len(M[i])):
            M[i][j] = round(M[i][j],decPts)

# 1.4 计算矩阵的转置
# https://docs.python.org/dev/tutorial/controlflow.html#unpacking-argument-lists
# http://python3-cookbook.readthedocs.io/zh_CN/latest/c04/p11_iterate_over_multiple_sequences_simultaneously.html
# 用* -operator写入函数调用以将参数从列表或元组中解开
def transpose(M):
    return [list(col) for col in zip(*M)]

# 1.5 计算矩阵乘法 AB，如果无法相乘则raise ValueError
# http://www.jb51.net/article/68532.htm 
def matxMultiply(A, B):
    try:
        if len(A[0]) != len(B):
            raise ValueError
        return [[sum(a * b for a, b in zip(a, b)) for b in zip(*B)] for a in A]
    except ValueError:
        raise ValueError('Two length are not eq.')

# 2.1 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    return [a + b for a,b in zip(A,b)]

# 2.2 
# 初等行变换
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# 把某行乘以一个非零常数
def scaleRow(M, r, scale):
    if scale != 0:
        for i,row in enumerate(M[r]):
            M[r][i] = M[r][i] * scale
    else:
        raise ValueError('scale cannot be 0.')

# 把某行加上另一行的若干倍
def addScaledRow(M, r1, r2, scale):
    M[r1] = [x + y * scale for x,y in zip(M[r1],M[r2])]

# 2.3 Gaussian Jordan 消元法求解 Ax = b

# 2.3.2 算数步骤详见ipynb

# 2.3.3 实现 Gaussian Jordan 消元法

# TODO 不知道怎么找到最大值所在的index

# https://rosettacode.org/wiki/Reduced_row_echelon_form
def gj_Solve(A,b,decPts=4, epsilon = 1.0e-16):
    # 构造增广矩阵
    M = augmentMatrix(A, b)

    lead = 0
    rowCount = len(M)
    columnCount = len(M[0]) - 1

    # 如果行与列数量不等,返回None
    if rowCount != columnCount:
        return None

    for r in range(rowCount):
        if columnCount <= lead:
            return None
        
        # 我还是不会……
        maxIdx = r
        for row in range(r+1, rowCount):
            if Ab[row, r] > Ab[maxIdx, r]:
                maxIdx = row
            

        while M[maxIdx][lead] == 0:
            maxIdx = maxIdx + 1
            if rowCount == maxIdx:
                maxIdx = r
                lead = lead + 1
                if columnCount == lead:
                    break
        

        swapRows(M,r,maxIdx)

        if M[r][lead] is not 0:
            scaleRow(M, r, M[r][lead])

        for maxIdx in range(rowCount):
            if 0 <= maxIdx and maxIdx < rowCount:
                if maxIdx != r:
                   addScaledRow(M,maxIdx,r,-M[maxIdx][lead])
        
        lead = lead + 1

    # 取出最后一列
    result = [[row[-1]] for row in M]

    # 四舍五入
    matxRound(result, decPts)

    return result
    

# 3 线性回归

# 3.1 随机生成样本点 详见ipynb

# 3.2 拟合一条直线

# 3.2.1 猜测一条直线 详见ipynb

# 3.2.2 计算平均平方误差 (MSE)
def calculateMSE(X,Y,m,b):
    mx = []
    len_of_line = len(X)
    y = []
    MSE = 0
    for i in enumerate(X):
        mx.append(i[1])
     for j,num in enumerate(Y):
        MSE += ( (num - mx[j] - b) * (num - mx[j] - b) )

    MSE = MSE/len_of_line
    
    return MSE

# print(calculateMSE(X,Y,m1,b1))

# 3.4 求解  XTXh=XTY
# 差一个gj_Solve就能出结果
def linearRegression(X,Y):
    x = []
    y = []
    for i,r in enumerate(X):
        x.append([r,1])
    for i,r in enumerate(Y):
        y.append([r])
    XT = transpose(x)
    A = matxMultiply(XT, x)
    b = matxMultiply(XT, y)
    result_list = gj_Solve(A, b)
    return result_list[0][0], result_list[1][0]

# m2,b2 = linearRegression(X,Y)
# assert isinstance(m2,float),"m is not a float"
# assert isinstance(b2,float),"b is not a float"
# print(m2,b2)