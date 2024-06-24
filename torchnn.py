# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
# Adam is optimizer
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
#  dependencies to convert images to tensors
from torchvision.transforms import ToTensor

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# Converting data into batches of 32 images
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network (subclass of neural network)
class ImageClassifier(nn.Module): 
    def __init__(self):
        # subclass the model
        super().__init__()
        #using sequential API from PyTorch
        self.model = nn.Sequential(
            #images are black and white so input channel of one, 32 images that are 3 by 3
            nn.Conv2d(1, 32, (3,3)), 
            # activation 
            nn.ReLU(),
            # 32 as input and out put 64
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            # 64 as input and 64 output
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            # falttening the layer in one dimention
            nn.Flatten(), 
            # Shaving off 2 pxel on each level so 6 in general
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
# Instance of neural network sending to cuda GPT
clf = ImageClassifier().to('cuda')
# instantiate optimizer and specify learning rate
opt = Adam(clf.parameters(), lr=1e-3)
# loss function creationg
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    for epoch in range(10): # train for 10 epochs
        for batch in dataset: # looping through datasey
            X,y = batch 
            X, y = X.to('cuda'), y.to('cuda') # to cpu or gpu
            yhat = clf(X) # make prediction
            loss = loss_fn(yhat, y) #calculate loss

            # Apply backprop 
            opt.zero_grad() # zero out any existing gradients
            loss.backward() # calculate gradients
            opt.step() # apply gradient decent

        print(f"Epoch:{epoch} loss is {loss.item()}") # print loss for every batch
    ## save the model to our environment
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 
    # Loading the weights into our classifier
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  
    # import image
    img = Image.open('img_3.jpg') 
    # converting image to tensor
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    #
    print(torch.argmax(clf(img_tensor)))