from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, Sigmoid

# def define_model(num_features, num_filter, drop):
#     model = Sequential()
#     model.add_module('conv1', Conv2d(1, num_filter, kernel_size=(1, num_features)))
#     model.add_module('relu1', ReLU())
#     model.add_module('conv2', Conv2d(num_filter, num_filter, kernel_size=(3, 1)))
#     model.add_module('relu2', ReLU())
#     model.add_module('pool1', MaxPool2d(kernel_size=(2, 1)))
#     model.add_module('conv3', Conv2d(num_filter, num_filter, kernel_size=(3, 1)))
#     model.add_module('relu3', ReLU())
#     model.add_module('pool2', MaxPool2d(kernel_size=(2, 1)))


class CNNpred(Module):
    def __init__(self, num_features, num_filter, drop):
        super(CNNpred, self).__init__()

        self.conv1 = Conv2d(1, num_filter, kernel_size=(1, num_features))
        self.relu1 = ReLU()
        self.conv2 = Conv2d(num_filter, num_filter, kernel_size=(3, 1))
        self.relu2 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=(2, 1))
        self.conv3 = Conv2d(num_filter, num_filter, kernel_size=(3, 1))
        self.relu3 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=(2, 1))
        self.drop1 = Dropout(drop)
        self.fc1 = Linear(104, 1)
        self.sig1 = Sigmoid()

    # Defining the forward pass    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.sig1(self.fc1(x))

        return x