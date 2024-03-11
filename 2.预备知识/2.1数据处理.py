import torch

# 1.入门
x = torch.arange(12)
print(x)
print(x.shape)              # 用shape参数，访问张量沿每个轴的长度的形状（shape）
print(x.numel())            # 用numel()，检查size（大小），来知道张量中元素总数

x = x.reshape(3,4)   # reshape返回一个修改了形状的新向量,可以用x.reshape_(3,4)或者x.resize_(3,4)直接修改原向量
print(x)
print()
# 但是也可以直接用-1来自动计算，例如：x = x.reshape(-1,4),等同于reshape(3,4)

# 直接用全0来初始化矩阵：
x = torch.zeros((2,3,4))
print(x,'全0','\n')
#直接用全1来初始化矩阵：
x = torch.ones((2,3,4))
print(x,'全1','\n')
# 构造数组来作为神经网络中的参数时，会随机初始化参数的值
x = torch.randn(3,4)
print(x,'随机','\n')

print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]))
# 外层列表应用于轴0，内层列表应用于轴1（Tips：在二维数组中：axis值0：第一个轴，即为行；axis值1：第二个轴。即为列）
print('\n\n')



# 2.运算符
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y,x-y,x*y,x/y,x**y)
print(torch.exp(x))             # 可以得到元素为e^x[i]的y[i]

# 将多个张量连接起来，并指出沿哪个轴连接
x = torch.arange(12,dtype=torch.float32).reshape(3,4)
y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0))           # dim=0，按照一行一行的顺序连接
print(torch.cat((x,y),dim=1))           # dim=1，按照一列一列的顺序连接

print(x==y)         # 相同则为True
print(x.sum())      # 求和
print()




# 3.广播机制
# 将不同形状的向量合并
a = torch.arange(3).reshape(3,1)
b = torch.arange(2).reshape(1,2)
print(a)
print(b)

# 由于a和b分别是3×1，1×2的矩阵，如果让它们相加，它们的形状不匹配。我没将两个矩阵广播为一个更大的3×2矩阵
# 做法：a复制列，b复制行
'''
a:[[0,1],    b:[[0,1],
   [0,1],       [0,1],
   [0,1]]       [0,1]]
'''
print(a+b)
print()



# 4.索引和切片
print(x)
# -1选择最后一个元素，1:3选择第二到第三个元素
print(x[-1],x[1:3])

#除读取外，还可以通过指定索引值修改元素来写入矩阵
x[1,2]=9
print(x)
#还可以为多个元素赋相同值，只需要索引所有元素，然后赋值
x[0:2,:]=2          # [0:2,:]表示访问第一行和第二行，其中“:”代表沿轴1（列）的所有元素
print(x)
print()



# 5.节省内存
before = id(y)
y = y + x
print(id(y)==before)
# 用id函数获取地址，发现执行y=y+x，其实是给y重新分配了一个内存地址，因此为了节约内存，要原地执行，使用切片法将操作的结果分配给先前分配的数组，例如y[:] = <expression>
# 为了说明这一点,建立一个新的矩阵z，其形状与y相同，使用zeros_like来分配一个 全0的块
z = torch.zeros_like(y)
print(z)
print('id(z):',id(z))
z[:] = x + y
print(z)
print('id(z):',id(z))

# 如果后续计算中没有重复使用x，可以使用x[:] = x + y 或 x += y来减少操作的内存开销
before = id(x)
x += y
print(id(x)==before)



