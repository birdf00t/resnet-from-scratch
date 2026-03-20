import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data_func
from ResNet import ResNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#데이터 없으면 다운로드 있으면 불러오기
trainloader, testloader = data_func.get_dataloaders()

model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
for epoch in range(10):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        print(f'epoch {epoch} loss: {loss.item()}')


correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'정확도: {100 * correct / total:.2f}%')