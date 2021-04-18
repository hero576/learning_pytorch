import os

import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import LeNet


class Predictor:
    def __init__(self, weight_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        assert os.path.exists(weight_path)
        self.net = LeNet().to(device=self.device)
        self.net.load_state_dict(torch.load(weight_path))

    def predict(self,img):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        img = self.transform(image).to(device=self.device)
        img = torch.unsqueeze(img,0)
        with torch.no_grad():
            outputs = self.net(img)
            print(outputs)
            outputs = torch.softmax(outputs,dim=1)
            print(outputs)
            index = torch.argmax(outputs,dim=1)
            return index

if __name__ == '__main__':
    p = Predictor("weights/LeNet_21.pth")
    image_path = r"E:\Users\gm\Pictures\2.png"
    img = cv2.imread(image_path)
    res = p.predict(img)
    print(res)


