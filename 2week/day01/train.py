from net import MNIST_Net
from data import MNIST_Dataset
#创建训练类
class Train:
    def __init__(self,root):
        self.train_dataset=MNIST_Dataset(root,)