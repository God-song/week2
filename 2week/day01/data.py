#自定义数据集
#测试机验证模型训练结果的
#训练集合用于模型训练学习的
import numpy as np
#DataLoader数据加载器
from torch.utils.data import Dataset,DataLoader
import os
import cv2
class MNIST_Dataset(Dataset):
    #初始化函数
    def __init__(self,root,is_train=True):
        #之所以存放的是图片路径而不是图片是因为，图片内存很大，浪费空间时间
        self.dataset=[]
        sub_dir="TRAIN" if is_train else "TEST"
        #print(f"{root}/{sub_dir}")
        #显示./MNIST_IMG/MNIST_IMG/TRAIN
        for tag in os.listdir(f"{root}/{sub_dir}"):
            img_dir =f"{root}/{sub_dir}/{tag}"
            #显示./MNIST_IMG/MNIST_IMG/TRAIN/0-1-2-3-4
            #print(img_dir)
            for img_filename in os.listdir(img_dir):
                #img_filename 那倒是的tag文件夹下面的所有照片，图片名称
                img_path=f"{img_dir}/{img_filename}"
                #获得图片的路径
                #print(img_path)
                #将所有图片和文件名加到dataset
                self.dataset.append((img_path,tag))

    #返回数据集的长度
    def __len__(self):
        #返回所有数据长度
        return len(self.dataset)
    #每条数据的处理方式
    def __getitem__(self, item):
        #获取索引为item的数据
        #重要
        data=self.dataset[item]
        #拿到图片路径，加载图片进来,[0]是图片地址
        #[1]是该图片文件名，也是答案

        #读成单通道图片
        img_data=cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)
        #img_Data是一个二位数组，28*28，里面的数字是0-255
        #是一个numpy类型,[28 28]形状
        #现在我希望输入的是一个[28*28]降维，以下两种办法的都已
        img_data=img_data.reshape(28*28)
        #img_data=img_data.reshape(-1)
        #-1自动去算
        #现在得到的是一个拥有784个数字的一维数据
        #不能够直接往神经网络里面输入
        #需要做归一化处理，因为不做归一化处理，会因为
        #超出精度而报错，2数字大，计算量大，速度慢
        #对图片做归一行
        img_data=img_data/255

        #one -hot 编码,将文件名最为索引，将对应位置改成1
        tag_one_hot=np.zeros(10)
        tag_one_hot[int(data[1])]=1
        print(tag_one_hot)
        #由于神经网络的float是32位的这边改一下位数
        return np.float32(img_data),np.float32(tag_one_hot)

if __name__=="__main__":
    dataset=MNIST_Dataset("./MNIST_IMG/MNIST_IMG")
    # for tag in os.listdir("./MNIST_IMG/MNIST_IMG/TEST"):
    #     print(tag)
    #print(dataset.__getitem__(0))
    #一次取多少张，是否打乱
    dataloader=DataLoader(dataset,batch_size=10,shuffle=True)