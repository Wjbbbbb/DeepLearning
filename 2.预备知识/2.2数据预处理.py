import os

# 1.读取数据集
# 举个例子，先创建一个人工数据集，并存储在CSV(逗号分隔值)文件../data/house_tiny.csv中。
os.makedirs(os.path.join('..','data'),exist_ok=True)            # exist_ok=True，表示如果文件夹存在，。不会报错
data_file = os.path.join('..','data','house_tiny.csv')          # path.join()中的,分隔多个字符，构成一个../data/house_tiny.csv路径
with open(data_file,'w') as f:                                  # 打开文件，以w模式进入，文件句柄被赋值给变量f
    f.write('NumRooms,Alley,Price\n')       # 列名
    f.write('NA,Pave,127500\n')             # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    # 最后with open块结束后，文件将会关闭

# 要从创建的CSV文件中加载原始数据集，导入pandas包并调用read_csv函数。该数据集有4行3列，其中每行描述了房间数量(NumRooms)、巷子类型(Alley)和房屋(Price)。
import pandas as pd
data = pd.read_csv(data_file)
print(data)




# 2.处理缺失值
# 注意，NaN项代表缺失值，处理缺失的数据的典型方法包括插值法(用一个代替值弥补缺失值)和删除法(直接忽略缺失值)。
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]                # iloc进行位置索引。iloc是基于整数位置进行选择和访问数据的方法。
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs.iloc[2:4,0].mean())
print(inputs)

# 对于inputs中的类别值或离散值，我们将NaN视为一个类别。由于Alley列只接受两种类型的类别值Pave和NaN，pandas可以自动将阿此列转换为两列Alley_pave和Aleey_nan.
# Alley列为Pave的行回将Alley_Pave的值设置为1，Alley_nan的值设置为0.缺失Alley列的行会将Alley_Pave和Alley_nan分别设置为0和1。
inputs = pd.get_dummies(inputs,dummy_na=True).astype(int)           # 不加astype(int)，得到的结果是True和False，加上得到的是1，0
print(inputs)
print()



# 3.转换为张量格式
# 现在inputs和outputs中的所有条目都是数值类型，他们可以转换为张量格式。当数据采用张量格式后，可以通过2.1中引入的张量函数来进一步操作
import torch
x,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(x)
print(y)