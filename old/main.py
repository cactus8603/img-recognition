from cProfile import label
from operator import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import net

import matplotlib.pyplot as plt

from keras.datasets import mnist

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

def train(train_loader, test_loader, epoch):
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []

    model = net()
    input_shape = (-1,1,28,28)
    lr = 0.01
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
 
    

    for epochs in range(epoch):
        correct_train = 0
        total_train = 0

        for i, (image, labels) in enumerate(train_loader):
            train = image.view(input_shape)
            labels = labels

            opt.zero_grad()
            output = model(train)

            t_loss = criterion(output, labels)
            t_loss.backward()

            opt.step()
            predicted = torch.max(output.data, 1)[1]
            total_train += len(labels)
            
            correct_train += (predicted==labels).float().sum()

        acc = 100 * correct_train / float(total_train)
        train_acc.append(acc)
        train_loss.append(t_loss.item())

        correct_test = 0
        total_test = 0

        for i, (image, labels) in enumerate(test_loader):
            test = image.view(input_shape)

            output = model(test)
            v_loss = criterion(output, labels.long())
  
            predicted = torch.max(output.data, 1)[1]
            total_test += len(labels)
            correct_test += (predicted==labels).float().sum()

        val_acc = 100 * correct_test / float(total_test)
        valid_acc.append(val_acc)
        valid_loss.append(v_loss.item())

        print('Train Epoch:{}/{} Training_loss:{} Training_acc:{:.6f}% val_loss:{} val_acc:{:.6f}%'.format(epochs, epoch, t_loss.data, acc, v_loss.data, val_acc))
    return train_loss, train_acc, valid_loss, valid_acc

if __name__ == '__main__':

   

    # in mnist use this method otherwise use dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    train_data, test_data, target_train, target_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)

    target_train = torch.from_numpy(target_train)
    target_test = torch.from_numpy(target_test).type(torch.Tensor)

    TRAIN = TensorDataset(train_data, target_train)
    TEST = TensorDataset(test_data, target_test)

    lr = 0.01
    batch = 100
    iters = 10000
    epochs = int(iters/(len(train_data)/batch))


    train_loader = DataLoader(TRAIN, batch_size=batch, shuffle=True)
    test_loader = DataLoader(TEST, batch_size=batch, shuffle=True)

    training_loss, training_accuracy, validation_loss, validation_accuracy = train(train_loader, test_loader, epochs)

    # visualization
    plt.plot(range(epochs), training_loss, 'b-', label='Training_loss')
    plt.plot(range(epochs), validation_loss, 'g-', label='validation_loss')
    plt.title('Training & Validation loss')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(range(epochs), training_accuracy, 'b-', label='Training_accuracy')
    plt.plot(range(epochs), validation_accuracy, 'g-', label='Validation_accuracy')
    plt.title('Training & Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

