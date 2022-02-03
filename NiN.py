import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    nin_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU())
    return nin_block

net = nn.Sequential(
    nin_block(1,96,kernel_size=11,strides=4,padding=0),nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(96,256,kernel_size=5,strides=1,padding=2),nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256,384,kernel_size=3,strides=1,padding=1),nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(p=0.5),
    nin_block(384,10,kernel_size=3,strides=1,padding=1),
    nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())

print(net)
X = torch.randn(1,1,224,224)
for layer in net:
    X = layer(X)
    print("{}outputshape:\t{}".format(layer.__class__.__name__,X.shape))


batch_size = 128
train_iter,test_iter = load_fashion_mnist(batch_size , resize = 224)
num_epochs , lr = 5 , 0.1

# 借助d2l包训练并可视化结果
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
d2l.plt.show()


loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr)

net = net.cuda()
def init_weight(m):
    if type(m)==nn.Conv2d:
        nn.init.xavier_normal_(m.weight)
net.apply(init_weight)

for epoch in range(num_epochs):
    for X ,y in train_iter:
        net.train()
        optim.zero_grad()
        X ,y = X.cuda(),y.cuda()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
print(d2l.evaluate_accuracy_gpu(net,train_iter,"cuda:0"),d2l.evaluate_accuracy_gpu(net,test_iter,"cuda:0"))

