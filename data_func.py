import torch
import torchvision
import torchvision.transforms as transforms   
import os

def _check_data():
    #데이터 파일 다운 여부 체크
    return os.path.exists("data/cifar-10-batches-py")

def _del_zip():
    #만약 필요없는 zip이 존재한다면 삭제
    tar_path = "data/cifar-10-python.tar.gz"
    if os.path.exists(tar_path):
        os.remove(tar_path)
    return

def get_dataloaders(batch_size=128 ):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    download=False
    if (not _check_data()):
        download=True

    trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=download, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=download, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    _del_zip()

    print(f"훈련 데이터: {len(trainset)}장")
    print(f"테스트 데이터: {len(testset)}장")
    
    return trainloader, testloader

