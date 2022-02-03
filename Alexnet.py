import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l

# Alexnet
net = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Flatten(),
    nn.Linear(256*5*5,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,10))

# 测试
X = torch.randn(1,1,224,224)
for layer in net:
    X = layer(X)
    print(f'{layer.__class__.__name__} output shape:\t{X.shape}')


batch_size = 128
train_iter,test_iter = load_fashion_mnist(batch_size , resize= 224)
num_epochs , lr = 5 , 0.1

# 借助d2l包训练并可视化结果
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
d2l.plt.show()


# 手动训练
net.cuda()
def init_weight(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight)
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr)
for epoch in range(num_epochs):
    for X, y in train_iter:
        net.train()
        optim.zero_grad()
        X , y = X.cuda() , y.cuda()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
print("训练集：{}\n测试集：{}".format(d2l.evaluate_accuracy_gpu(net,train_iter,"cuda:0"),d2l.evaluate_accuracy_gpu(net,test_iter,"cuda:0")))
