# 1.标量
import torch
print('1.')
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y,x * y,x / y,x**y)
print()



# 2.向量
print('2.')
x = torch.arange(4)
print(x)
# 通常认为，列向量是向量的默认方向，这里也如此
print(x[3])
# 长度。维度，形状
print(len(x))
print(x.shape)
print()



# 3.矩阵
# 正如向量将标量从零阶推广到一维，矩阵将向量从一维推广到二维，在代码表示为具有两个轴的张量。
# 在调用函数来实例化张量时，我们可以通过指定两个分量m和n来创建一个形状为m×n的矩阵
print('3.')
A = torch.arange(20).reshape(5,4)
print(A)
print(A[1,2])       # 访问某一个元素
print(A.T)          # 转置
# 定义一个对称矩阵
B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B == B.T)



# 4.张量
# 向量是一阶张量，矩阵是二阶张量
print('4.')
X = torch.arange(24).reshape(2,3,4)
print(X)
print()


# 5.张量算法的基本性质
print('5.')
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()
print(A,A+B)
print()
print(A*B)                  # a11*b11 a12*b12 ... a1n*b1n
print()


# 6.降维
print('6.')
x = torch.arange(4,dtype=torch.float32)
print(x)
print(x.sum())
# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变成一个标量，可以指定沿着哪个轴降低
A = torch.arange(20,dtype=torch.float32).reshape(5,4)
print(A)

A_sum_axis0 = A.sum(axis=0)             # 加完成为一个沿着行的向量
print(A_sum_axis0)
print(A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)

# 沿着axis=0,1的方向求和，等价于矩阵中的所有元素求和
print(A.sum(axis=[0,1])==A.sum())

print(A.mean())             # 平均值
print(A.sum()/A.numel())
# 同样计算平均值的函数也可以沿指定轴降低张量的维度
print(A.mean(axis=0))
print(A.shape)
print(A.sum(axis=0)/A.shape[0])


# 7.点积
print('7.')
x = torch.arange(4,dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))

# 8.矩阵-向量积 (martix-vector)
print('8.')
# A^T X
print(A,x,torch.mv(A,x))

# 9.矩阵-矩阵乘法 (martix-martix)
print('9.')
B = torch.ones(4,3)
print(torch.mm(A,B))

# 10.范数     ||x||
print('10.')
# 范数来L2=根号下(Xi ^2)，L1=绝对值(Xi)
u = torch.tensor([3.0,-4.0])
print(torch.norm(u))


