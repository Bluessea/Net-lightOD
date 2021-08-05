import sys
sys.path.append('./data')
sys.path.append('./model')

import torch
import torchvision.transforms as transforms
from model.backbones.mobilenetv3_blockResidual import MobileNetV3_Large
from model.backbones.mobilenetv3_blockResidual import MobileNetV3_Small
from PIL import Image

# 创建一个检测器类，包含了图片的读取，检测等方法
class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self,net_kind,num_classes=17):
        super(Detector, self).__init__()
        kind=net_kind.lower()
        if kind=='large':
            self.net = MobileNetV3_Large(num_classes=num_classes)
        elif kind=='small':
            self.net = MobileNetV3_Small(num_classes=num_classes)
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def load_weights(self,weight_path):
        self.net.load_state_dict(torch.load(weight_path,map_location="cpu"))

    # 检测器主体
    def detect(self,weight_path,pic_path):
        # 先加载权重
        self.load_weights(weight_path=weight_path)
        # 读取图片
        img=Image.open(pic_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor=img_tensor.cuda()
        net_output = self.net(img_tensor)
        print(net_output)
        _, predicted = torch.max(net_output.data, 1)
        result = predicted[0].item()
        print("预测的结果为：",result)

if __name__=='__main__':
    detector=Detector('large',num_classes=17)
    detector.detect('./weight/best.pkl','./test.jpg')







