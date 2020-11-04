from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as tvmodel
import torch.nn as nn
import torch
import visdom
import time
import matplotlib.pyplot as plt
import pdb
from torchsummary import summary
# root_dir = 'I:/workspace/greedy/homework/homework7/VOCtrainval_06-Nov-2007/VOCdevkit/'
root_dir=os.getcwd()+'/VOCtrainval_06-Nov-2007/VOCdevkit/'
# pdb.set_trace()
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
inputsize = 448
NUM_BBOX=2

def convert(size,box):
    """将bbox的左上角和右下角坐标格式转换为bbox中心点+bbox的w，h格式并归一化"""
    dw=1./size[0]
    dh=1./size[1]
    x=(box[0]+box[1])/2.0
    y=(box[2]+box[3])/2.0
    w=box[1]-box[0]
    h=box[3]-box[2]
    x=x*dw#归一化操作
    w=w*dw
    y=y*dh
    h=h*dh
    return(x,y,w,h)
def convert_xml(file_path, out_file):
    #将xml文件转换为txt文件
    out_file = open(out_file, 'w')
    tree = ET.parse(file_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

#         bb = (max(1, float(xmlbox.find('xmin').text)), max(1, float(xmlbox.find('ymin').text))
#               , min(w - 1, float(xmlbox.find('xmax').text)), min(h - 1, float(xmlbox.find('ymax').text)))

        points=(float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
        bb = convert((w,h), points)

        out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb])+'\n')
    out_file.close()

sets = [('2007', 'train'), ('2007', 'val')]
for data_ in sets:
    if not os.path.exists(root_dir + 'VOC%s/Labels/' % (data_[0])):
        os.makedirs(root_dir + 'VOC%s/Labels/' % (data_[0]))
    name_list = open(root_dir + 'VOC%s/ImageSets/Main/%s.txt' % (data_[0], data_[1])).read().strip().split()

    print(len(name_list))
    name_list = tqdm(name_list)
    data_list = open('VOC%s_%s.txt' % (data_[0], data_[1]), 'w')

    file_writer = ''
    for i, xml_name in enumerate(name_list):
        file_path = root_dir + 'VOC%s/Annotations/%s.xml' % (data_[0], xml_name)
        label_file = root_dir + 'VOC%s/Labels/%s.txt' % (data_[0], xml_name)
        img_file = root_dir + 'VOC%s/JPEGImages/%s.jpg' % (data_[0], xml_name)
        convert_xml(file_path, label_file)

        file_writer += img_file + ' ' + label_file + '\n'

    data_list.write(file_writer)
    file_writer = ''

    data_list.close()


def show_labels_image(imgname):
    """imgname是输入的图像的名称，无下标"""
    import cv2
    img = cv2.imread(root_dir + 'VOC2007/JPEGImages/' + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w, h)
    print(img.shape)
    label = []
    with open(root_dir + "VOC2007/Labels/" + imgname + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(classes[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img, classes[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow("img", img)
    cv2.waitKey(0)


show_labels_image('000007')

def convert_bbox2labels(bbox):
    """
    bbox 含有多个obj，一个obj有5个值（cls,x,y,w,h），bbox为一维数组或列表
    x,y为归一化的值，转换后的x,y为bbox中心点在每个网格中相对网格的位置
    """

    """将bbox的（cls，x,y,w,h）转换为训练时方便计算loss的数据形式（7，7，5*B+cls_num）
    输入的是（xc，yc，w，h）格式，输出labels后bbox信息转换为了（px，py，w，h）的格式"""
    gridsize=1.0/7
    labels=np.zeros((7,7,5*NUM_BBOX+len(classes)))
    '''
    YOLO直接将全图划分为S×S的网格，每个格子负责中心在该格子上的
    目标检测，一次性预测所有格子所含的目标
    YOLO网络结构是由24个卷积层、2个全连接层构成，网络入口448×448，图片进入网络，先
   进行resize，网络的输出结果为一个张量维度为S×S(B*5+C)。其中S为划分的网格数量，B是
    每个网格负责目标个数，C是类别个数，5代表(x,y,w,h,置信度)。论文中B=2，C=20。

    '''

    for i in range(len(bbox)//5):#标签中含有多个obj，一个obj有5个值
        #print(i)
        gridx=int(bbox[i*5+1]//gridsize)   #当前bbox中心在网格第gridx列
        gridy=int(bbox[i*5+2]//gridsize)   #当前bbox中心在网格第gridy行
        #gridx,  gridy为归一化数据
        gridpx=bbox[i*5+1]/gridsize-gridx  #当前bbox中心相对位置
        gridpy=bbox[i*5+2]/gridsize-gridy
        # gridpx,  gridpy为bbox中心点在每个网格中相对网格的位置
        # pdb.set_trace()
        labels[gridy,gridx,0:5]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        labels[gridy,gridx,5:10]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        labels[gridy,gridx,10+int(bbox[i*5])]=1  #class的种类标签
    return labels


def calculate_iou(bbox1, bbox2):
    intersect_bbox = [0., 0., 0., 0., ]
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]:
        return 0  # 没有相交区域，或者return 0？
    else:
        intersect_bbox[0] = max(bbox1[0], bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])
    #    if area_intersect>0:
    return area_intersect / (area1 + area2 - area_intersect)
#    else:
#        return 0

import torchvision.transforms as transforms


#数据加载器
class MYDATA():
    def __init__(self, is_train=True, is_aug=True):
        self.filenames = []  # 存储数据集的文件名称
        if is_train:
            with open(root_dir + "VOC2007/ImageSets/Main/train.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        else:
            with open(root_dir + "VOC2007/ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.imgpath = root_dir + "VOC2007/JPEGImages/"
        self.labelpath = root_dir + "VOC2007/Labels/"
        self.is_aug = is_aug

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        img = cv2.imread(self.imgpath + self.filenames[item] + ".jpg")
        h, w = img.shape[:2]
        input_size = inputsize
        padw, padh = 0, 0
        if h > w:
            padw = (h - w) // 2 #如果h比w大，则在w上进行填充
            #
            img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
        elif w > h:
            padh = (w - h) // 2
            img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)  # img(375,500,3)

        img = cv2.resize(img, (input_size, input_size))

        # 转换为张量处理
        if self.is_aug:
            #把PIL.Image或者numpy.narray数据类型转变为torch.FloatTensor类型，shape是C*H*W，数值范围缩小为[0.0, 1.0]。
            # transforms.ToTensor()做了类型转化及归一化处理；compose（）对transform列表进行遍历
            aug = transforms.Compose([transforms.ToTensor()])
            img = aug(img)

        # 读取图片对应的bbox信息，按照1维方式存储，每五个元素表示为一个bbox的（cls，xc，yc，w，h）
        #bbox为规一化的值
        with open(self.labelpath + self.filenames[item] + ".txt") as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        # pdb.set_trace()
        bbox = [float(x) for y in bbox for x in y]
        # pdb.set_trace()
        if len(bbox) % 5 != 0:
            raise ValueError("File:" + self.labelpath + self.filenames[item] + ".txt" + "bbox extraction error!")

        for i in range(len(bbox) // 5):  # 得到pod后的（x,y,w,h）
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w

        labels = convert_bbox2labels(bbox)
        labels = transforms.ToTensor()(labels)
        return img, labels


class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1, self).__init__()

    def forward(self, pred, labels):
        num_gridx, num_gridy = labels.size()[-2:]
        # pdb.set_trace()
        #        num_b=2
        #        num_cls=5
        #         print(num_gridx,num_gridy)   #7,7
        noobj_confi_loss = 0. #无目标置信度损失
        coor_loss = 0. #坐标损失
        obj_confi_loss = 0. #有目标置信度损失
        class_loss = 0.
        n = labels.size()[0]  # batchsize的大小
        for i in range(n):
            for m in range(7):
                for n in range(7):
                    if labels[i, 4, m, n] == 1:  # 如果包含物体
                        """将数据（px，py，w，h）转换为（x1，y1，x2，y2）
                        先将px，py转换为cx，cy，即相对网格位置转换为标准化后实际bbox的中心位置cx，cy
                        再利用（cx-w/2,cy-h/2,cx+w/2,cy+h/2）转换为xyxy的形式，用于计算iou
                        """
                        bbox1_pred_xyxy = ((pred[i, 0, m, n] + m) / num_gridx - pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy - pred[i, 3, m, n] / 2,
                                           (pred[i, 0, m, n] + m) / num_gridx + pred[i, 2, m, n] / 2,
                                           (pred[i, 1, m, n] + n) / num_gridy + pred[i, 3, m, n] / 2)
                        # pdb.set_trace()
                        bbox2_pred_xyxy = ((pred[i, 5, m, n] + m) / num_gridx - pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / num_gridy - pred[i, 8, m, n] / 2,
                                           (pred[i, 5, m, n] + m) / num_gridx + pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / num_gridy + pred[i, 8, m, n] / 2)
                        bbox_gt_xyxy = ((labels[i, 0, m, n] + m) / num_gridx - labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + n) / num_gridy - labels[i, 3, m, n] / 2,
                                        (labels[i, 0, m, n] + m) / num_gridx + labels[i, 2, m, n] / 2,
                                        (labels[i, 1, m, n] + n) / num_gridy + labels[i, 3, m, n] / 2)
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        if iou1 > iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 4, m, n] - iou1) ** 2  # pred[i,4,m,n] 置信度
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 9, m, n] - iou2) ** 2  # pred[i,4,m,n]
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n] - iou1) ** 2)

                        class_loss = class_loss + torch.sum((pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    else:
                        # 降低没有物体时，判断为有物体的损失
                        noobj_confi_loss = noobj_confi_loss + 0.5 * (
                                    torch.sum(pred[i, 4, m, n] ** 2) + torch.sum(pred[i, 9, m, n] ** 2))

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        return loss / n

class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet,self).__init__()
        resnet = tvmodel.resnet18(pretrained=True)  # 调用torchvision里的resnet预训练模型,pretrained=True，表示使用预训练模型初始参数
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层：平均池化层和全连接层，前面都是卷积层（加了跳跃连）
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            #Torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride=1，padding=0，dilation=1，groups=1，bias=True)
            #in_channels：输入维度；out_channels：输出维度； kernel_size：卷积核大小；stride：步长大小；padding：填充；dilation：kernel间距
            nn.Conv2d(resnet_out_channel,512,3,padding=1),     #1024
            nn.BatchNorm2d(512),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(512,512,3,stride=2,padding=1), #此处再缩小尺寸一倍
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*512,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,7*7*30),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)#前面多维度的tensor展平成一维
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(classes)), 7, 7)  # 记住最后要reshape一下输出数据


# epoch=80
epoch = 1
batchsize = 8
lr = 0.0001
train_data = MYDATA()
train_dataloader = torch.utils.data.DataLoader(MYDATA(is_train=True), batch_size=batchsize, shuffle=True)
model = YOLOv1_resnet().cuda()
for layer in model.children():
    layer.requirs_grad = True
    break
criterion = Loss_yolov1()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

Loss_list_train = []


for e in range(epoch):
    model.train()
    train_loss = 0.0
    y1=torch.Tensor([0]).cuda()
    start = time.time()
    # print(len( train_dataloader))
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs=inputs.cuda()
        print("read item done!")
        # print(i)
        labels = labels.float().cuda()
        pred = model(inputs)
        loss = criterion(pred, labels)
        train_loss += loss.item()
        optimizer.zero_grad()  # 梯度置0
        loss.backward()
        optimizer.step()  # 更新模型
        print("Epoch %d/%d| Step %d/%d| Loss: %.2f" % (e, epoch, i, len(train_data) // batchsize, loss))
        #            y1=y1+loss
        #            if is_vis: #and (i+1)%100==0:
        #                vis.line(np.array([y1.cpu().item()/(i+1)]),np.array([i+e*len(train_data)//batchsize]),win=viswin1,update='append')
        end = time.time()
        print('Train time: {:.6f}'.format(end - start))
        start = time.time()
    Loss_list_train.append(train_loss / len(train_dataloader))

    if e > 10 and (e + 1) % 10 == 0:
        torch.save(model, root_dir + "VOC2007/Models/YOLOv1_epoch" + str(e + 1) + ".pkl")
