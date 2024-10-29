
import torch
from IPython import display
from d2l import torch as d2l

#这里定义一个实用程序类Accumulator，用于对多个变量进行累加。 在evaluate_accuracy函数中， 
# 我们在Accumulator实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 
# 当我们遍历数据集时，两者都将随着时间的推移而累加
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
# 一个用来展示变化动画的类
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 预测值和真实值比较，得到正确判断的数量，这里的预测值是原始值，就是softmax的结果
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1: # 如果大于一行且大于1列
        y_hat = y_hat.argmax(axis=1) # 每一行都得到最大值的那个列下标，比如第一行最大值在第三列，结果就是y_hat[0]=3，意思是第一个样本预测值为3
    cmp = y_hat.type(y.dtype)==y  #y_hat转成y的数据类型，并且比较两个向量，得到一个都是 true,false组成的张量
    return float(cmp.type(y.dtype).sum())  # 先转类型为数字类型，true代表1，sum()会把所有true求和

# 计算在指定数据集的模型精度
def evaluate_accuracy(net,data_inter):
    if isinstance(net,torch.nn.Module): # 如果模型是用的pytorch的，需要设置评估模式
        net.eval()                      # 所谓评估模式就是 witchNoGrad不计算梯度，只做前向
    metric = Accumulator(2)
    for X,y in data_inter:
        metric.add(accuracy(net(X),y),y.numel()) # 第一个是准确个数，第二个是张量个数，也就是标签个数，代表数据个数
    return metric[0]/metric[1]          # 返回正确率

# 每一批次的训练过程
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)                     # 存放三个变量
    for X,y in train_iter:
        y_hat = net(X)                          # 得到softmax结果
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()                 # 清空梯度
            l=l.mean()
            l.backward()                        # 反向传播 计算w和b的梯度，框架的loss函数不需要总和
            updater.step()                      # 对w和b进行更新，使用pytorch的模型和优化器，优化器会自动找到模型的w和b,net[0].weight和批量大小等信息
            metric.add(
                float(l)*len(y)                 # 存储总损失值
                ,accuracy(y_hat,y)              # 存储总正确个数
                ,y.size().numel()               # 存储总标签数，size() 返回的 torch.Size 对象拥有 numel() 方法，计算张量形状的所有元素的乘积。而直接调用 y.numel() 也是完全等效的
            )
        else:
            l.sum().backward()                  
            updater(X.shape[0])                 # 根据批量大小update
            metric.add(
                float(l.sum())                  # 存储总损失值
                ,accuracy(y_hat,y)              # 存储总正确个数
                ,y.size().numel()               # 存储总标签数，size() 返回的 torch.Size 对象拥有 numel() 方法，计算张量形状的所有元素的乘积。而直接调用 y.numel() 也是完全等效的
            )
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 总体训练函数
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    # 每一批次对应准确度动画
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater) # 得到每一批次的各个参数，并且更新了模型权重
        # 看看更新后的模型在测试数据集的准确度
        test_acc = evaluate_accuracy(net,test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics   # 最后一批次结束后的最终损失值和准确度
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc