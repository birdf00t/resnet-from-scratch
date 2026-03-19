import data_func

#데이터 없으면 다운로드 있으면 불러오기
trainloader, testloader = data_func.get_dataloaders()