import torch
import torchvision


class BasicClassif(torch.nn.Module):
    def __init__(self):
        super(BasicClassif, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.dp = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dp(x)
        x = self.fc2(x)
        return x
    
def torch_info():
    print('Torch version:', torch.__version__)
    print('Torch is build with CUDA:', torch.cuda.is_available())

def gpu_info():
    gpu_number = torch.cuda.device_count()
    print('GPU is available:', gpu_number > 0)
    if gpu_number > 0:
        print('Torch GPU count:', gpu_number)
        for ind in range(gpu_number):
            print("*"*20)
            print("GPU {}:".format(ind))
            print("Device name:", torch.cuda.get_device_name(ind))
            print("Compute capability:", torch.cuda.get_device_capability(ind))
            print("Memory:", torch.cuda.get_device_properties(ind).total_memory/1024/1024, "MB")


def get_dataloaders(download=True, batch_size=64, num_workers=0, pin_memory=False):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=download,
                       transform=transform)
    test_dataset = torchvision.datasets.MNIST('./data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=pin_memory)
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.sum(loss_fn(output, target)).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":


    torch_info()
    gpu_info()

    print()

    train_loader, test_loader = get_dataloaders(download=True,
                                                    batch_size=64,
                                                    num_workers=0,
                                                    pin_memory=False)

    model = BasicClassif().to("cuda")
    loss = torch.nn.CrossEntropyLoss().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 6):
        train(model, torch.device('cuda'), train_loader, optimizer, loss, epoch)
        test(model, torch.device('cuda'), test_loader, loss)
