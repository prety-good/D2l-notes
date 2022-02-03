import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l

# Lenet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),nn.Sigmoid(),nn.AvgPool2d(2),
    nn.Conv2d(6, 16, kernel_size=5),nn.Sigmoid(),nn.AvgPool2d(2),
    nn.Flatten(),
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10))

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)


# 读取fashion-mnist数据集
batch_size = 64
train_iter,test_iter = load_fashion_mnist(batch_size)

num_epochs , lr = 10,0.9

# 借助d2l包训练并可视化结果
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
d2l.plt.show()




def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def init_weights(m):
    if type(m)==nn.Conv2d or type(m) == nn.Conv2d:
        # nn.init.normal_(m.weight,0,0.1)
        nn.init.xavier_normal_(m.weight)
net.apply(init_weights)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr)

net.cuda()
for epoch in range(num_epochs):
    net.train()
    for X,y in train_iter:
        optim.zero_grad()
        X=X.cuda()
        y=y.cuda()
        y_hat = net(X)
        l = loss(y_hat,y)
        l.sum().backward()
        optim.step()
print("训练集：{}\n测试集：{}".format(evaluate_accuracy_gpu(net,train_iter,"cuda:0"),evaluate_accuracy_gpu(net,test_iter,"cuda:0")))