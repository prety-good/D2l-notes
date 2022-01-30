***
# D2l笔记
***

## 3.在服务器上演示

* 服务器演示 安装build-essentential  python3.8 conda conda的环境管理 jupyter d2l torch torchvision  rise插件

* ssh -L8888:locallhost:8888 root@ip&emsp;映射远端的8888（jupyter）的端口到本地的8888的端口

* colab sagemaker:免费的GPU资源

***

## 4.数据操作

* n维数组 标量 向量  矩阵    &emsp;rgb图片为3为数组 &emsp; 一个批量的图片 4维 视频 5维的

* 切片 取子域

* 张量即为一个多维数组

* `torch.arange()`&emsp;`tensor.numel()`&emsp;`tensor.reshape()`&emsp;`zeros()` `ones()`&emsp;`torch.tensor([list])`

* numpy和torch数据转换：`tensor.numpy()` `torch.tensor(ndarray)`

* 张量的基本运算：+、-、*、/、**对应元素的运算&emsp;`torch.exp()`&emsp;`torch.cat((X,Y),dim=x)`按照第x维进行拼接

* 广播机制&emsp;列向量和行向量进行运算

* numpy:ndarray&emsp;torch:tensor&emsp;tensor.item()对单个元素的tensor(标量)转化为数字

* `os.makedirs()`&emsp;`os.path.join()`&emsp;`f.write()`

* pandas&emsp;`pd.read_csv()`

* 处理缺失的数据：插值和删除

  插值：`df.fillna(df.mean())`均值替代缺失值

  离散值独热编码`pd.get_dummies(df,dummy_na=True)`代表对缺失值也进行编码

* 接着把处理好的df类型数据转化为tensor

***

## 5.线性代数

* 标量&emsp;向量（内积即为点乘)&emsp;矩阵  (范数)&emsp;

* 实现：`len(tensor)``tensor.size()``tensor.T`

  向量是标量的推广，矩阵是向量的推广&emsp;`clone()`深拷贝

* `sum(axis=x)`按照第x个维度进行求和&emsp;keepdims保留维度，有利于进行广播运算

* `torch.dot(a,b)`点积，等价于`torch.sum(a*b)` 

* `torch.mm(a,b)`矩阵乘矩阵 &emsp;`torch.mv(X,b)`矩阵乘向量

  `torch.matmul(X,Y)`等价于`torch.mv()`和`torch.mm()`的集合&emsp;其实也等价于`np.matmul(a,b)`

  `torch.dot(a,b)`向量的内积 等价于`np.dot(a,b)`

  矩阵的对应元素相乘直接使用a*b

* 向量的L2范数`torch.norm()`:平方和再开根号&emsp;向量的L1范数`torch.abs(tensor).sum()`:绝对值求和&emsp;向量的Lp范数即为$\|X\|_p= \left({\sum_{i=1}^{n}{|x_i|^p}}\right)^{1/p}$&emsp;矩阵的F范数`torch.norm()`

* 特定轴求和

***

## 6.矩阵计算

* 函数导数的链式法则

***

## 7.自动微分

* 向量的链式法则手动求导

  符号求导&emsp;数值求导&emsp;自动求导

* 计算图&emsp;链式法则&emsp;前向传播&emsp;反向传递

  - 前向传播时获得所有的计算结果并保存到内存中，接着反向传播，反向传播计算顺序与前向传播相反。

  - 反向传递需要前向传播的所有中间结果，故深度学习中显存的要求比较大。这也是容量较大的模型可能会导致内存不足(out of memory)的原因之一。

* 实现：`x.requires_grad_(True)`表明需要保存grad，存放在`x.grad`中。`y=2*x`即隐式地构造了计算图，再使用`y.backward()`即可进行反向传递，x的梯度就保存到了`x.grad`中。`x.grad.zero_()`清空梯度。

* 将计算移出计算图：`detach()`视为常数来计算。

* 显示构造：先使用占位符构建模型，然后填入数据进行计算。&emsp;隐式构造：直接赋值，在复制计算过程中构建计算图。

* 损失函数loss值通常为标量，多个loss时需要累计梯度

***

## 8.线性回归

* 简单的线性模型，定义损失函数，求得参数w、b来使得损失函数最小。

* 线性模型有显示解，损失函数为平方损失。

* 梯度下降&emsp;W(t)=W(t-1)-l对W(t-1)的导数

  小批量的随机梯度下降：选取随机b个样本的loss值作为近似，小批量随机梯度下降是深度学习默认的求解算法，超参数：批量大小b、学习率learing rate

* 手动实现线性回归：

  手动生成数据集`torch.normal()`&emsp;定义数据的`dataloder`

  手动构建线性模型&emsp;设置`requires_grad=True`

  定义损失函数、优化算法

  训练：计算损失`loss()`，反向传播`backward()`，参数优化，计算准确率

* 框架实现：  

  `nn.Sequential()` `nn.Linear(in,out)`

  损失函数：`nn.MSEloss()`优化器：`torch.optim.SGD(net.parameters,lr=x)`传入所有需要更新的参数和learing rate

***

## 9.softmax回归

* 是一个分类问题&emsp;分类问题与回归问题 多分类 可以获得所有类别的置信度 和为1

  $\hat{y}$：预测的y

* 损失函数： 均方损失 L2loss：$l=\frac{1}{2}(y_1-y_2)^2$

  L1loss:$l=|y-y'|$

* 图像分类数据集

* softmax回归手动实现：

  `loss=nn.CrossEntropyloss()`
  
* fashionMNIST数据集：导入data包：`from torch.utils import data`。&emsp;把训练集转换为TensorDataset：`dataset=data.TensorDataset(*(X,y))`，如果是mnist数据集的话需要使用torch的函数`mnist_train=torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor())`。&emsp;定义迭代器DataLoader：`data.DataLoader(dataset,batch_size,shuffle=True,num_workers=1)`得到一个可迭代的对象，循环遍历一遍数据集作为一个训练的epoch。

***

## 10.多层感知机

* 收敛定理：在$\frac{r^2+1}{\rho^2}$步后收敛

* 多层感知机&emsp;输入层，隐藏层，输出层

  相对于线性模型多了激活函数，激活函数是非线性，成为了神经网络。

* 激活函数：必须是非线性的

  * Sigmoid:$\frac{1}{1+e^{-x}}$ 范围(0,1)`torch.sigmoid()`
  * Tanh:$\frac{1-e^{-2x}}{1+e^{-2x}}$ 范围(-1,1)`torch.tanh()`
  * ReLU:${max(x,0)}$ 范围(0,正无穷)`torch.relu()`最常用

* 多类分类：使用softmax，在pytorch中使用`nn.CrossEntropyLoss()`损失函数，集成了softmax层。

  该交叉熵损失函数的第一个参数为各标签的概率，第二个参数为真实标签，故应该`l=loss(y_hat,y)`，不可颠倒。

* 多隐藏层：最好是逐渐地缩小隐藏层的参数

* 多层感知机的手动实现：

  `torch.zeros_like()`

***

## 11.模型选择、过拟合和欠拟合

* 泛化误差和训练误差

* 训练集、验证集和测试集

  在数据集较小的时候使用K折交叉验证

* 欠拟合(underfitting)和过拟合(overfitting)

* 模型容量：拟合各种函数的能力

  - 低容量模型难以拟合复杂数据，欠拟合

  - 高容量模型会记住所有简单的训练数据，导致过拟合
  - 训练误差随模型容量增大而减小，泛化误差会在一定容量的模型时出现一个最优值

* 统计学习的概念：VC维

  - VC维：对于VC个数据，无论怎么样改变标签，总存在一个模型可以完成完美预测。

  - n维线性模型的VC维等于n+1

* 数据复杂度：样本个数、单个样本的元素个数、时间和空间结构、多样性

* 模型容量和数据的复杂度需要匹配，否则可能会导致欠拟合或过拟合。

***

## 12.权重衰退 weight_decay

* weight_decay

* L2正则化： 在loss函数上加上正则化项。

  pytorch中在optim中加上weight_decay：`torch.optim.Adam(,lr=,weight_decay=1e-3)`

***

## 13.丢弃法 Dropout

* 丢弃法：作用在隐藏层的输出上面：`h'=dropout(h,p)`

  将一些（概率为p的）隐藏层的输出置为0来控制模型的复杂度。

***

## 14.数值稳定性、模型初始化和激活函数

* 数值稳定性：从理论上验证了梯度消失和梯度爆炸的原因和后果。
  - 梯度爆炸：会导致参数更新过大，破坏模型的稳定收敛
  - 梯度消失：会导致参数更新过小，模型几乎无法学习，无法达到收敛。

* 让训练更加稳定：目标：让梯度在合理范围内。如：[1e-6,1e3]

  - 将乘法变加法：ResNet，LSTM

  - 归一化：梯度归一化、梯度裁剪
  - 使用合理的权重初始值和合理的激活函数

* 权重的初始化：相同的初始值无法打破对称性，多个同样权重的单元好像只有一个单元。无法实现网络的表达能力。

  - Xavier初始化

  * 远离最优解的地方，损失函数表面可能很复杂。在最优解附近，损失函数一般会比较平滑。
  * 使用N(0,0.01)来初始化对小网络可以使用，但是不能保证适用于深度神经网络。

* 激活函数的选取：

  - sigmoid：$\frac{1}{1+e^{-x}}$&emsp;由图像可以看出，当输入的x过大或者过小时都会导致梯度接近于0，在多层的网络中，很可能会在某一层切断梯度导数梯度消失，导致模型训练速度缓慢甚至无法收敛。
  - tanh
  - relu 较常用

* 环境和分布偏移、协变量偏移

  分布偏移是指训练集和测试集有可能来自不同的分布。在现实问题中，可能会有环境导致的因素被误认为是特征，从而导致模型的错误。

> *推理公式太复杂没看懂*  

***

## 15.实战Kaggle房价预测

* 

>

***

## 16.Pytorch基础

* **块**的定义：可以描述单个层、多个层组成的组件或整个模型。

  代码角度来看，一个块就是一个类(class)，每个块都需要一个$forward$前向传播函数，并保存有必要的参数。

* `nn.Sequential()`

  通常直接调用`net(X)`实际上是`net.__call__(X)`的简写，在nn.Module中，该函数被定义为了执行`forward()`函数，故继承$nn.Module$的类的前向传播函数都应命名为$forward()$

* 自定义的块应该为`nn.Module`的子类，借此可以完成自己定义的网络，实现Sequential满足不了的计算，更加灵活。

* 参数管理：`net.state_dict()`：状态，对于线性层包括了weight和bias。&emsp;`net.bias` `net.bias.data`直接访问参数。

* 参数初始化:`nn.init`中有许多用于初始化的函数，如`nn.init.normal_(net.weight,mean,std)`,`nn.init.zeros_(net.bias)`,`nn.init.constant_(X,b)`,`nn.init.uniform_(X,-,-)`

* 特殊用法：参数绑定、参数共享。可以实现在不同的网络间共享权重。

* 自定义层：参数需要放在`self.weight=nn.Parameter(···)`中

* 读写文件：
  - 存取变量`torch.save(a,'dic')`&emsp;`a=torch.load('dic')`
  - 存取网络的参数`torch.save(net.state_dict(),'···')` 取的时候`net.load_state_dice(torch.load('···'))`在pytorch中网络的定义无法保存，需要重新定义。这也是pytorch相较于tenserflow和mxnet不足的地方，无法保存网络的定义。

***

## 17.使用GPU

* 查看gpu驱动是否正常：`!nvidia-smi`

* 访问device：`torch.device('cpu')` `torch.device('cuda:0')`

  查看数量`torch.cuda.device_count()`

* 查看变量位置`x.device`    变更位置：`x=x.cuda(0)` `x=x.cpu()`

* 移动网络：`net = net.to('cuda:0')`

***

## 18.卷积层

* 对于二维的图像数据而言，全连接网络忽略掉了图像的空间信息，而卷积神经网络能够提取并处理空间结构信息。

* 两个原则：平移不变性&emsp;局部性

* 全连接层到卷进层的转变过程：

  对于全连接层：

  - 应用到图像后，输入变为了矩阵，权重变为了4维：$h_{i,j} = \sum_{a,b}     v_{i,j,a,b}  x_{i+a,j+b}$

  - 由于平移不变性，对于不同部位的权重应该相同。故4维的权重去掉了两维重复的部分。$$h_{i,j} = \sum_{a,b}    v_{a,b} x_{i+a,j+b}$$
  - 由于局部性：对于输入，只关注局部的一部分，故每次运算的时候只需要对输入的一部分进行运算。这就是2维卷积(2维交叉相关)。$$h_{i,j} = \sum_{a=-\Delta}^{\Delta}\sum_{b=-\Delta}^{\Delta}   v_{a,b} x_{i+a,j+b}$$
  - 输入：$n_h*n_w$输出:$\left( n_h-k_h+1 \right)*\left( n_w-k_w+1 \right)$

* 一维卷积：$y_i=\sum_{a=1}^{h}w_ax_{i+a}$可以处理文本、语言、时序序列信息。

  三维卷积：$$y_{i,j,k} = \sum_{a=1}^{h}\sum_{b=1}^{w}\sum_{c=1}^{d}   w_{a,b,c} x_{i+a,j+b,k+c}$$用于处理视频、医学图像、气象地图信息、深度图像。

***

## 19.填充和步幅 padding and stride

* 填充 padding:(此处的padding是上下一共填充的大小，比如周围全部加上一排0，padding即为$1*2=2$)&emsp;一般取为k-1，即周围填充$\frac{k-1}{2}$，使得尺寸大小不变

  输入：$n_h*n_w$输出:$\left( n_h-k_h+p_h+1 \right)*\left( n_w-k_w+p_w+1 \right)$

* 步幅 stride：一般取1，缩小图像时取2使得卷积后尺寸缩小一半

  输入：$n_h*n_w$输出:$ \lfloor\frac{n_h-k_h+p_h+s_h }{s_h}\rfloor*\lfloor\frac{ n_w-k_w+p_w+s_w }{s_w}\rfloor$

* 通过padding，把网络做深，小的$kernal$的感受野也会越来越大，最终覆盖整个输入的图片。深的网络+小的$kernal$相较于浅的网络+大的$kernal$效果更好。

***

## 20.卷积中的多通道

* RGB图片：三通道图片，200x200的图片尺寸：$200*200*3$

* 输入：$c_i*n_h*n_w$ &emsp;卷积核：$c_i*k_h*k_w$&emsp;输出：$m_h*m_w$

  对于每个输入通道，都有一个卷积核，最终多个输入通道的卷积结果相加变为一个输出通道。

  输入通道和每个卷积层的卷积核的个数相同，输出通道和卷积层的个数相同。

* 1x1卷积层：它不识别空间模式，只是融合通道。

* 二维卷积层的总结：

  - 输入：$X:c_i*n_h*n_w$
  - 核： $W:c_o*c_i*k_h*k_w$
  - 偏差bias：$B:c_o*c_i$
  - 输出 $Y：c_o*m_h*m_w$

***

## 21.池化层

* pooling：一般为最大池化，有padding和stride，输入通道数=输出通道数。
* 最大池化输出的是窗口中的最强的模式信号。平均池化层相较更加柔和。
* Pytorch中的池化层的步幅大小和窗口大小默认相同
* 池化层放在卷积层的输出之后，可以使得结果对于位置的信息不是那么的敏感。
