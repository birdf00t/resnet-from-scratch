import matplotlib.pyplot as plt
import data_func
from ResNet import ResNet


#데이터 없으면 다운로드 있으면 불러오기
trainloader, testloader = data_func.get_dataloaders()
images, labels = next(iter(trainloader))

rb = ResNet()
output = rb.forward(images)
print(output.shape)




