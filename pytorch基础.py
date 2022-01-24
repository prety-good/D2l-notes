import torch
from torch import device, nn 


'''
层和块
'''
# 定义多层感知机

net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(20,256)
        self.f2 = nn.Linear(256,10)

    def forward(self, x):
        return self.f2(torch.relu(self.f1(x)))
# 实例化类：
mlp = mlp()

x = torch.rand(2,20)
print('x:{}\nnn.Sequential:{}\nmlp:{}'.format(x,net(x),mlp(x)))


# 顺序块
class Mysequential(nn.Module):
    def __init__(self , *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self , x):
        for block in self._modules.values():
            x=block(x)
        return x

net=Mysequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print('Mysequential:',net(x))


# 嵌套使用
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU())
        self.net1 = Mysequential(nn.Linear(128,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net1(self.net(X)))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 8))
print(f'嵌套使用{chimera(x)}\n')



'''
参数管理
'''
net = nn.Sequential(nn.Linear(2,5),nn.ReLU(),nn.Linear(5,1))
# 参数访问
print(net,net[2].state_dict(),type(net[2].weight),net[2].weight,net[2].weight.data,net[2].bias,net[2].bias.data, sep='\n')
print( net.state_dict()['2.bias'])

# 嵌套块
block = nn.Sequential(nn.Linear(2,5),nn.ReLU(),nn.Linear(5,5),nn.ReLU())
net = nn.Sequential(block,block,nn.Linear(5,1))
print(net , net[0][0].state_dict() ,net[0][0].weight ,sep='\n')

# 参数初始化
def init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,100,0.01)
        nn.init.constant_(m.bias,11)
net.apply(init)
print('初始化：',net[0][0].state_dict())

net[0][0].weight.data[:]=666
print('初始化：',net[0][0].state_dict())

# 参数绑定
share = nn.Linear(10,10)
net = nn.Sequential( nn.Linear(2,10),nn.ReLU(),share,nn.ReLU(),share,nn.ReLU(),nn.Linear(10,2))
print(net[2].weight == net[4].weight)




'''
自定义层
'''

# 不带参数的层
class layer1(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x-x.mean()
x = torch.rand(2,5)
layer1 = layer1()
print(f'不带参数的层：{layer1(x)},均值：{layer1(x).mean():f}')

# 带参数的层
class layer2(nn.Module):
    def __init__(self, input , out):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(input,out) )
        self.bias = nn.Parameter(torch.zeros(out))
    def forward(self,x):
        return torch.relu(torch.matmul(x,self.weight.data) +self.bias.data)
linear = layer2(5,2)
print('带参数的层：',linear.state_dict(),linear(x),sep='\n')


'''
读写文件
'''
# 读写张量
torch.save(x,'tmp/x_file')
y=torch.load('tmp/x_file')
print(x==y)
mydict = {'x': x, 'y': y}
torch.save(mydict, 'tmp/mydict')
mydict2 = torch.load('tmp/mydict')
print(mydict2)

# 读写网络参数
torch.save(linear.state_dict(),'tmp/layer2')
layer22=layer2(5,2)
layer22.load_state_dict(torch.load('tmp/layer2'))
print (linear(x) ==layer22(x))


'''
使用GPU
'''

print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count()>i:
        return torch.device(f'cuda:{i}')
    else:
         return torch.device('cpu')

x = torch.rand(1,3,device = try_gpu())
print(x,x.device)
x = x.cpu()
print(x,x.device)
x = x.cuda(0)
print(x,x.device)


layer22 = layer22.to(try_gpu())
print(layer22.weight.data.device)
layer22 = layer22.to('cpu')
print(layer22.weight.data.device)
