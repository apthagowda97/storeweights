import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from storeweights import weights


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = TestModel()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epoch = 20

weights.save("test_model", model, optimizer, extra_info={"epoch": epoch},gdrive=True)
weights.save("test_model", model, optimizer, extra_info={"epoch": epoch})

weights.load("test_model", model, optimizer, return_extra_info=True)
weights.load("test_model", model, optimizer, version=1, return_extra_info=True)

weights.remove("test_model", version=2)
weights.remove("test_model")

weights.save("test_model", model, optimizer, extra_info={"epoch": epoch})

weights.view("test_model")
