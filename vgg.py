import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l

# vgg的块
def vgg_block(conv_num , cin, cout):
    block = []
    for _ in range(conv_num):
        block.append(nn.Conv2d(cin,cout,kernel_size=3,padding=1))
        block.append(nn.ReLU())
        cin = cout
    block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)
net = vgg_block(3,1,64)
print(f'vgg_block测试：\n{net}')


def vgg(conv_arch):
    blks = []
    cin = 1
    for conv_num,cout in conv_arch:
        blks.append(vgg_block(conv_num, cin ,cout))
        cin = cout
    net = nn.Sequential(
        *blks,
        nn.Flatten(),
        nn.Linear(cout * 7 * 7,4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
    return net


conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))
net = vgg(conv_arch)
print(f'vgg测试：\n{net}')
X = torch.randn(1,1,224,224)
for layer in net:
    X = layer(X)
    print("{}outputshape:\t{}".format(layer.__class__.__name__,X.shape))



batch_size = 128
train_iter,test_iter = load_fashion_mnist(batch_size , resize = 224)
num_epochs , lr = 10 , 0.1

# 借助d2l包训练并可视化结果
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
d2l.plt.show()

net.cuda()
def init_weight(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        nn.init.xavier_normal_(m.weight)
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr)
for epoch in range(num_epochs):
    for X,y in train_iter:
        net.train()
        optim.zero_grad()
        X , y = X.cuda(),y.cuda()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
print(d2l.evaluate_accuracy_gpu(net,train_iter,"cuda:0"),d2l.evaluate_accuracy_gpu(net,test_iter,"cuda:0"))

