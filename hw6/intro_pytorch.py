import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_set = datasets.FashionMNIST('./ data', train = True, download = True, transform = transform)
    test_set = datasets.FashionMNIST('./ data', train = False, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
    if training:
        return train_loader
    else:
        return test_loader



def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model




def train_model(model, train_loader, criterion, T):
    model = model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):

        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            correct = correct + torch.sum(outputs.argmax(1) == labels)
        accuracy = (100. * correct / len(train_loader.dataset))
        print(f'Train Epoch: {epoch}  Accuracy: {int(correct)}/{len(train_loader.dataset)}({float(accuracy):.2f}%) Loss: {running_loss/len(train_loader):.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = False):
    model = model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if show_loss:
        print(f'Average loss: {running_loss/len(test_loader):.4f}')
    print(f'Accuracy: {100 * correct / total:.2f}%')


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    outputs = model(test_images)
    prob = F.softmax(outputs, dim=0)
    prob_list = prob[index].tolist()
    top_three = sorted(zip(prob_list, class_names), reverse=True)[:3]
    print(f'{top_three[0][1]}: {100 * top_three[0][0]:.2f}%')
    print(f'{top_three[1][1]}: {100 * top_three[1][0]:.2f}%')
    print(f'{top_three[2][1]}: {100 * top_three[2][0]:.2f}%')



if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss=True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)