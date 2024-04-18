#Deep_Learning #Programming 

```python
# Lab 11 MNIST and Deep learning CNN
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True)

# CNN Model


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self._build_net()

    def _build_net(self):
        # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
        self.keep_prob = 0.7
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.keep_prob = 0.5
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform(self.fc2.weight)

        # define cost/loss & optimizer
        self.criterion = torch.nn.CrossEntropyLoss()    
        # Softmax is internally computed.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        self.eval()
        # cuda gpu 에 데이터 전달
        x = x.to(device)
        return self.forward(x)

    def get_accuracy(self, x, y):
        # cuda gpu 에 데이터 전달
        x, y = x.to(device), y.to(device)
        prediction = self.predict(x)
        correct_prediction = (torch.max(prediction.data, 1)[1] == y.data)
        self.accuracy = correct_prediction.float().mean()
        return self.accuracy

    def train_model(self, x, y):
        self.train()
        x, y = x.to(device), y.to(device)
        self.optimizer.zero_grad()
        hypothesis = self.forward(x)
        self.cost = self.criterion(hypothesis, y)
        self.cost.backward()
        self.optimizer.step()
        return self.cost


# instantiate CNN model
model = CNN().to(device)

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(mnist_train) // batch_size

    for i, (batch_xs, batch_ys) in enumerate(data_loader):
        X = Variable(batch_xs)    # image is already size of (28x28), no reshape
        Y = Variable(batch_ys)    # label is not one-hot encoded

        cost = model.train_model(X, Y)

        avg_cost += cost.data / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))

print('Learning Finished!')

# Test model and check accuracy
X_test = Variable(mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float())
Y_test = Variable(mnist_test.test_labels)

print('Accuracy:', model.get_accuracy(X_test, Y_test))   
```

위는 CNN model 을 정의한 것이다.
# 1. model.named_children()
자신 바로 아래의 <mark style='background:#f7b731'>자식만</mark> 반환
```python
for name, child in model.named_children():
    print(f"[ Name ] : {name}\n[ Children ]\n{child}")
    print("-" * 30)
```

**결과창**
```
[ Name ] : layer1
[ Children ]
Sequential(
  (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : layer2
[ Children ]
Sequential(
  (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : layer3
[ Children ]
Sequential(
  (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : fc1
[ Children ]
Linear(in_features=2048, out_features=625, bias=True)
------------------------------
[ Name ] : layer4
[ Children ]
Sequential(
  (0): Linear(in_features=2048, out_features=625, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
)
------------------------------
[ Name ] : fc2
[ Children ]
Linear(in_features=625, out_features=10, bias=True)
------------------------------
[ Name ] : criterion
[ Children ]
CrossEntropyLoss()
------------------------------
```
# 2. model.named_modules()
자신을 포함한 <mark style='background:#f7b731'>모든 submodule</mark> 출력

```python
for name, module in model.named_modules():
    print(f"[ Name ] : {name}\n[ Module ]\n{module}")
    print("-" * 30)
```

**결과창**
```
[ Name ] : 
[ Module ]
CNN(
  (layer1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.30000000000000004, inplace=False)
  )
  (layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.30000000000000004, inplace=False)
  )
  (layer3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.30000000000000004, inplace=False)
  )
  (fc1): Linear(in_features=2048, out_features=625, bias=True)
  (layer4): Sequential(
    (0): Linear(in_features=2048, out_features=625, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
  )
  (fc2): Linear(in_features=625, out_features=10, bias=True)
  (criterion): CrossEntropyLoss()
)
------------------------------
[ Name ] : layer1
[ Module ]
Sequential(
  (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : layer1.0
[ Module ]
Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
------------------------------
[ Name ] : layer1.1
[ Module ]
ReLU()
------------------------------
[ Name ] : layer1.2
[ Module ]
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
------------------------------
[ Name ] : layer1.3
[ Module ]
Dropout(p=0.30000000000000004, inplace=False)
------------------------------
[ Name ] : layer2
[ Module ]
Sequential(
  (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : layer2.0
[ Module ]
Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
------------------------------
[ Name ] : layer2.1
[ Module ]
ReLU()
------------------------------
[ Name ] : layer2.2
[ Module ]
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
------------------------------
[ Name ] : layer2.3
[ Module ]
Dropout(p=0.30000000000000004, inplace=False)
------------------------------
[ Name ] : layer3
[ Module ]
Sequential(
  (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
  (3): Dropout(p=0.30000000000000004, inplace=False)
)
------------------------------
[ Name ] : layer3.0
[ Module ]
Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
------------------------------
[ Name ] : layer3.1
[ Module ]
ReLU()
------------------------------
[ Name ] : layer3.2
[ Module ]
MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
------------------------------
[ Name ] : layer3.3
[ Module ]
Dropout(p=0.30000000000000004, inplace=False)
------------------------------
[ Name ] : fc1
[ Module ]
Linear(in_features=2048, out_features=625, bias=True)
------------------------------
[ Name ] : layer4
[ Module ]
Sequential(
  (0): Linear(in_features=2048, out_features=625, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
)
------------------------------
[ Name ] : layer4.1
[ Module ]
ReLU()
------------------------------
[ Name ] : layer4.2
[ Module ]
Dropout(p=0.5, inplace=False)
------------------------------
[ Name ] : fc2
[ Module ]
Linear(in_features=625, out_features=10, bias=True)
------------------------------
[ Name ] : criterion
[ Module ]
CrossEntropyLoss()
------------------------------
```