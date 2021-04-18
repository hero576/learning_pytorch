import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import LeNet


class Trainer:
    def __init__(self, batch_size=500, epochs=200, lr=0.001, save_path="../../weights".format(datetime.datetime.now())):
        # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        nw = 0
        self.epochs = epochs
        self.lr = lr
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = os.path.join(save_path, "LeNet_{}.pth")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_path = "../../data"
        assert os.path.exists(data_path)
        train_set = torchvision.datasets.MNIST(root=data_path, train=True, transform=self.transform, download=True)
        val_set = torchvision.datasets.MNIST(root=data_path, train=False, transform=self.transform, download=True)
        self.val_total = len(val_set)

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=nw)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=nw)

        self.net = LeNet().to(self.device)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def run(self):
        best_acc = 0.0
        for epoch in range(self.epochs):
            train_bar = tqdm(self.train_loader)
            running_loss = 0
            for step, data in enumerate(train_bar):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, self.epochs,
                                                                         running_loss / (step + 1))

            self.net.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(self.val_loader)
                for step, data in enumerate(val_bar):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.net(inputs)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, labels).sum().item()
                acc /= self.val_total
                print("valid epoch[{}/{}],acc={}".format(epoch + 1, self.epochs, acc))
                if best_acc < acc:
                    best_acc = acc
                    torch.save(self.net.state_dict(), self.save_path.format(epoch))


if __name__ == '__main__':
    t = Trainer()
    t.run()
