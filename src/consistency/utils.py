from collections import defaultdict
import torch
import torch.nn as nn


class CustomScheduler:
    '''
    Creates custom scheduler per epoch that can be used to weight
    different parameters such as loss weight or alpha. It's a linear
    function that oscilates between min and max values depending on 
    frequency provided.
    '''
    def __init__(self, min_val, max_val, num_epochs, freq):
        self.min_val = min_val
        self.max_val = max_val
        self.num_epochs = num_epochs
        self.freq = float(freq)
        self._current_val = self.min_val
        self.step_size = (self.max_val - self.min_val) * self.freq / num_epochs
        self.increasing = 1.0
        print('step size', self.step_size)

    def __call__(self, epoch):
        if epoch == 0:
            return self._current_val
        self._current_val += self.increasing * self.step_size
        if self._current_val >= self.max_val:
            self.increasing = -1.0
        elif self._current_val <= self.min_val:
            self.increasing = 1.0
        return max(self._current_val, 0.0)


class CifarCNN(nn.Module):
    '''
    Basic 2 Conv layer and 1 FC layer of Cifar CNN
    '''
    def __init__(self):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        x = self.fc1(x)
        return x


class SmallDataset(Dataset):
    '''
    Balanced Dataset that contains subset of larger dataset
    (not particularly efficient, just wrapper class to work 
    with predefined datasets such as cifar).
    '''
    def __init__(self, dataset, data_per_label = 5):
        super().__init__(self)
        data = defaultdict(list)
        i = 0
        while True:
            i += 1
            img, label = dataset[i]
            data[label].append(img)
            if min([len(v) for k, v in data.items()]) >= data_per_label:
                break
        for k in data:
            data[k] = data[k][:data_per_label]

        self.data = []
        for label, imgs in data.items():
            for img in imgs:
                img.to('cuda')
                self.data.append((img, label))

        print(f'len dataset: {len(self)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]


class ICTDataset(Dataset):
    '''
    Dataset for ICT approach that randomly samples another
    set of datapoints and returns 2 batches. No checking
    done to make sure the batches are exclusive.
    '''
    def __init__(self, dataset):
        super.__init__(self)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        ind2 = int(torch.randint(len(self), (1, 1)))
        inp1, label1 = self.dataset[ind]
        inp2, label2 = self.dataset[ind2]
        return inp1, inp2
