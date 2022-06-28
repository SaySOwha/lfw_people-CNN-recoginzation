import torch
import ssl
import warnings
import datetime
import pandas as pd
from torch import nn
from sklearn import datasets
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
batch_size = 32 # 批处理大小
learning_rate = 1e-2  # 学习率
epochs = 1000  # 训练次数
log_step_freq = 10

ssl._create_default_https_context = ssl._create_unverified_context
people = datasets.fetch_lfw_people(min_faces_per_person = 70)
images= people.images
labels=people.target
n = 7
img_x , img_y = 62,47
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.10)
x_train = x_train.reshape(x_train.shape[0],1, img_x, img_y)
x_test = x_test.reshape(x_test.shape[0],1, img_x, img_y)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train).long()
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test).long()

train_ids = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_ids, batch_size=batch_size, shuffle=True)
train_ids = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=train_ids, batch_size=8, shuffle=True)
print("data_loader completed")


class VGG(nn.Module):
    """
    VGG builder
    """
    def __init__(self, arch: object, num_classes,channels) -> object:
        super(VGG, self).__init__()
        self.in_channels = channels
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv3_256 = self.__make_layer(256, arch[2])
        self.conv3_512a = self.__make_layer(512, arch[3])
        self.conv3_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, 3, stride=1, padding=1, bias=False))  # same padding
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return F.softmax(self.fc3(out))

model = VGG([2, 2, 4, 4, 4], num_classes = n,channels = 1)
summary(model, input_size=(32, 1, img_x, img_y))



def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true,y_pred_cls)

model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
model.loss_func = nn.CrossEntropyLoss()
model.metric_func = accuracy
model.metric_name = "accuracy"


def train_step(model, features, labels):
    # 训练模式，dropout层发生作用
    model.train()
    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()


@torch.no_grad()
def valid_step(model, features, labels):
    model.eval()

    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()

def train_model(model,epochs,dl_train,dl_valid,log_step_freq):

    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns = ["epoch","loss",metric_name,"val_loss","val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("=========="*8 + "%s"%nowtime)

    for epoch in range(1,epochs+1):

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features,labels) in enumerate(dl_train, 1):
            loss,metric = train_step(model,features,labels)
            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_freq == 0:
                print(("[step = %d] loss: %.3f, "+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):

            val_loss,val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum/step, metric_sum/step,
                val_loss_sum/val_step, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f,"+ metric_name + \
              "  = %.3f, val_loss = %.3f, "+"val_"+ metric_name+" = %.3f")
              %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    return dfhistory

dfhistory = train_model(model,epochs,train_loader,test_loader,log_step_freq = log_step_freq)









