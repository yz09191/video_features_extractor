import os
import cv2
import numpy as np
import imageio
import skimage
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from IncepResv2 import InceptionResNetV2
from extract_frames import sample_frames
import torchvision


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_video(video_dir):
    # start = time.time()
    frames = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
    data = []
    for i, frame in enumerate(sorted(frames)):
        I = imageio.imread(os.path.join(video_dir, frame), mode='RGB')
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)
        I = (I.astype('float32') / 255.0 - 0.5) * 2
        data.append(I)
    if len(data) <= 0:
        return None
    res = np.asarray(data)[np.newaxis, :, :, :, :]
    # print("load time: ", time.time() - start)
    return res


def resize_frame(image, target_height=224, target_width=224):


    if len(image.shape) == 2:
        # 把单通道的灰度图复制三遍变成三通道的图片
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height), target_height))
        # print(resized_image.shape)
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        # print(cropping_length)
        resized_image = resized_image[:, cropping_length: target_width + cropping_length]
        # print(resized_image.shape)
    else:
        resized_image = cv2.resize(image, (target_width, int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length: target_height + cropping_length, :]

    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # 根据在ILSVRC数据集上的图像的均值（RGB格式）进行白化
    # image -= np.array([0.485, 0.456, 0.406])
    # image /= np.array([0.229, 0.224, 0.225])
    image -= np.array([0.5, 0.5, 0.5])
    image /= np.array([0.5, 0.5, 0.5])
    return image


def extract_features(video_path, aencoder, mencoder, max_frames, device):
    feature_size = 1536
    frame_list, clip_list = sample_frames(video_path, max_frames)
    # 把图像做一下处理，然后转换成（batch, channel, height, width）的格式
    frame_list = np.array([preprocess_frame(x, 299, 299) for x in frame_list])
    frame_list = frame_list.transpose((0, 3, 1, 2))
    # 先提取表观特征
    with torch.no_grad():
        frame_list = Variable(torch.from_numpy(frame_list)).to(device)
        print(frame_list.size())
        af = aencoder(frame_list)

    # 再提取动作特征
    clip_list = np.array([[resize_frame(x, 112, 112)
                           for x in clip] for clip in clip_list])
    clip_list = clip_list.transpose(0, 4, 1, 2, 3).astype(np.float32)
    with torch.no_grad():
        clip_list = Variable(torch.from_numpy(clip_list)).to(device)
        mf = mencoder(clip_list)

    # 视频特征的shape是max_frames x (2048 + 4096)
    # 如果帧的数量小于max_frames，则剩余的部分用0补足
    feats = np.zeros((max_frames, feature_size), dtype='float32')

    # 合并表观和动作特征
    # print(af.size(), mf.size())
    a_feats = af.data.cpu().numpy()
    m_feats = mf.data.cpu().numpy()
    # feats[:frame_count, :] = torch.cat([af, mf], dim=1).data.cpu().numpy()
    # feats[:frame_count, :] = af.data.cpu().numpy()
    return a_feats, m_feats


class AppearanceEncoder_inceptionresnetv2(nn.Module):
    def __init__(self):
        super(AppearanceEncoder_inceptionresnetv2, self).__init__()
        IRV2 = InceptionResNetV2(num_classes=1001)
        # print('IRV2:\n', IRV2)
        IRV2.load_state_dict(torch.load('F:/pretrained_models/inceptionresnetv2-520b38e4.pth'))
        modules = list(IRV2.children())[:-1]  # delete the last fc layer.
        self.IRV2 = nn.Sequential(*modules)
        # print('IRV2:\n', self.IRV2)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.IRV2(images)
        features = features.reshape(features.size(0), -1)
        # print(features.size())
        return features


class C3D(nn.Module):
    '''
    C3D model (https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py)
    '''

    def __init__(self):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(
            2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.dropout(F.relu(self.fc6(x)))
        x = self.dropout(F.relu(self.fc7(x)))

        return x


class MotionEncoder(nn.Module):

    def __init__(self):
        super(MotionEncoder, self).__init__()
        self.c3d = C3D()
        pretrained_dict = torch.load('F:/pretrained_models/c3d/ucf101-caffe.pth')
        model_dict = self.c3d.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.c3d.load_state_dict(model_dict)

    def forward(self, x):
        return self.c3d(x)


if __name__ == '__main__':
    print('device: ', DEVICE)
    appearance_extractor = 'resnet-101'
    motion_extractor = 'ResNeXt-101'


    print('Extracting 2D video feature by %s' % (appearance_extractor))
    if appearance_extractor == 'InceptionResnetv2':
        aencoder = AppearanceEncoder_inceptionresnetv2()
    elif appearance_extractor == 'resnet-101':
        aencoder = torchvision.models.resnet101()
        aencoder.load_state_dict(torch.load('F:/pretrained_models/resnet101-5d3b4d8f.pth'))
    aencoder.eval()
    aencoder.to(DEVICE)

    print('Extracting 3D video feature by %s' % (motion_extractor))
    if motion_extractor == 'C3D':
        mencoder = MotionEncoder()
    elif motion_extractor == 'ResNeXt-101':
        mencoder = torchvision.models.resnext101_32x8d()
        mencoder.load_state_dict((torch.load('F:/pretrained_models/resnext-101-kinetics.pth')))
    mencoder.eval()
    mencoder.to(DEVICE)

    video = 'video11.mp4'
    video_path = os.path.join('F:/dataset/MSR-VTT/Videos', video)
    a_feats, m_feats = extract_features(video_path, aencoder, mencoder, max_frames=26, device=DEVICE)
    print(a_feats.shape)
    print(m_feats.shape)

