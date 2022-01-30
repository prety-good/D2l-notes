import torch
from torch import nn

# 自定义卷积操作
def conv2d(X,K):
    h , w = K.shape
    Y = torch.zeros((X.shape[0]-h+1 , X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(conv2d(X, K))

# 自定义卷积层
class Conv2D(nn.Module):
    def __init__(self, ksize):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(ksize),requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1),requires_grad=True)
    def forward(self,X):
        return conv2d(X,self.weight)+self.bias

q=Conv2D((2,2))
print(q(X))
q.weight.data = K
print(q(X))
print(q.weight)

# 边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
print(conv2d(X, K),conv2d(X.T, K)) # 检测垂直边缘

# 卷积核参数的训练
net = Conv2D((1,2))
Y = conv2d(X,K)
lr = 2e-2

for i in range(10):
    y_hat = net(X)
    l = (y_hat-Y)**2
    l.sum().backward()
    net.weight.data[:] -= lr*net.weight.grad
    net.weight.grad.zero_()
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
print(net.weight.data)

net = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
lr = 3e-2
X=X.reshape(1,1,*(X.shape))
Y=Y.reshape(1,1,*(Y.shape))
for i in range(20):
    y_hat = net(X)
    l = (y_hat-Y)**2
    net.zero_grad()
    l.sum().backward()
    net.weight.data[:] -= lr*net.weight.grad
    if (i+1)%2 ==0:
        print(f'epoch {i+1},loss{l.sum():.3f}')
print(net.weight.data)


# padding & stride
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
X = torch.rand(size=(8, 8))
print(X.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)


def conv2d(X,K):
    h , w = K.shape
    Y = torch.zeros((X.shape[0]-h+1 , X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y
# 多输入通道
def corr2d_multi_in(X, K):
    return sum(conv2d(x,k) for x,k in zip(X,K))
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

# 多输出通道
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

K = torch.stack([K,K+1,K+2],0)
print(K.shape,corr2d_multi_in_out(X,K))


# 1x1卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
print(Y1==Y2)