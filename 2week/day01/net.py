#
import torch
from torch import nn
#创建一个mnist神经网络类，集成nn.Module模型基类
class MNIST_Net(nn.Module):
    """

    """
    #初始化函数
    def __init__(self):
        """
        初始化函数，调用父类初始化函数
        并且初始化，神经网络需要的参数
        """
        super(MNIST_Net, self).__init__()
        #nn.Sequential的含义是构建神经网络模块，在这里面
        # self.w1=torch.nn.Parameter(torch.randn(1,20))
        # self.b1=torch.nn.Parameter(torch.randn(20))
        # self.w2 = torch.nn.Parameter(torch.randn(20, 64))
        # self.b2 = torch.nn.Parameter(torch.randn(64))
        # self.w3 = torch.nn.Parameter(torch.randn(64, 128))
        # self.b3 = torch.nn.Parameter(torch.randn(128))
        # self.w4 = torch.nn.Parameter(torch.randn(128, 64))
        # self.b4 = torch.nn.Parameter(torch.randn(64))
        # self.w5 = torch.nn.Parameter(torch.randn(64, 1))
        # self.b5 = torch.nn.Parameter(torch.randn(1))

        # 简化，如何创建神经网络,定义网络层的方式
        # self.layer1=nn.Linear(1,20)
        # self.layer2 = nn.Linear(20, 64)
        # self.layer3 = nn.Linear(64, 128)
        # self.layer4 = nn.Linear(128, 64)
        # self.layer5 = nn.Linear(64, 1)
        # 再简化，定义网络可以将所有的网络层放在一起。进一步简化代码
        self.layer = nn.Sequential(
            #每一个这个Liner内部会自动创建b,偏移量，和w
            nn.Linear(28*28*1, 20),
            nn.Linear(20, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            #最后是一个十分类问题，最后输出是10
            nn.Linear(64, 10),
            #添加一个输出函数
            #里面要加一个维度，由于操作的是10所以维度为1
            #nn.Sigmoid
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        """
        神经网络的前向运算。
        :param x: 网络的输入值
        :return: 返回根据神经网络计算之后的值
        """
        # 叉乘
        # fc1=torch.matmul(x,self.w1)+self.b1
        # fc2=torch.matmul(fc1,self.w2)+self.b2
        # fc3 = torch.matmul(fc2, self.w3) + self.b3
        # fc4 = torch.matmul(fc3, self.w4) + self.b4
        # fc5 = torch.matmul(fc4, self.w5) + self.b5

        # 简化
        # fc1=self.layer1(x)
        # fc2 = self.layer2(fc1)
        # fc3 = self.layer3(fc2)
        # fc4 = self.layer4(fc3)
        # fc5 = self.layer5(fc4)

        #对于forward，多层神经网络应该是链式运算，每一层的输出是下一层的输入，最后将最后一层的输出当作结果
        #由于在构建神经网络，选择了比较简单的nn.Sequantial.所以返回比较简介
        out =self.layer(x)
        return out
#运算函数如果名字是__main__则是函数的主入口
if __name__=="__main__":
    #创建一个MNIST_Net实例对象
    net=MNIST_Net()
    #显示对象神经网络结构
    #print(net)
    #生成一个1x784的数据，测试一下
    # x=torch.randn(1,784)
    # out=net.forward(x)
    # print(out)
    # print(out.shape)
    #测试之后发现有问题，结果有正有负，需要利用输出函数，
    #控制输出值域
    #常用的输入函数有三个
    #liner,sigmoid(二分类常用 ,sortmax多分类常用

    #现在我们要利用这个神经网络模型，识别手写数字，但是要将输出数据处理一下