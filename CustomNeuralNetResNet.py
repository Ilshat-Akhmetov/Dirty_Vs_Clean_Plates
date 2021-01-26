import torch
from torchvision import models
class CustomNeuralNetResNet18(torch.nn.Module):
    def __init__(self,outputs_number):
        super(CustomNeuralNetResNet18, self).__init__()
        self.net = models.resnet18(pretrained=True)

        # Disable grad for all conv layers
        for param in self.net.parameters():
            param.requires_grad = True

        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, TransferModelOutputs // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 2, TransferModelOutputs // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 4, TransferModelOutputs // 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 8, outputs_number)
        )

    def forward(self, x):
        return self.net(x)
    #     self.net.fc=torch.nn.Linear(TransferModelOutputs,outputs_number)
    # def forward(self,x):
    #       x = self.net(x)
    #       return x

class CustomNeuralNetResNet50(torch.nn.Module):
    def __init__(self,outputs_number):
        super(CustomNeuralNetResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)

        # Disable grad for all conv layers
        for param in self.net.parameters():
            param.requires_grad = False

        TransferModelOutputs = self.net.fc.in_features
        self.net.fc = torch.nn.Sequential(
            torch.nn.Linear(TransferModelOutputs, TransferModelOutputs // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 2, TransferModelOutputs // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 4, TransferModelOutputs // 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(TransferModelOutputs // 8, outputs_number)
        )

    def forward(self, x):
        return self.net(x)
