import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import json

Learning_Rate=1e-4
width=height=900 # image width and height
batchSize=4

TrainFolder=r"/home/breakeroftime/Documents/Datasets/Transproteus/TranProteus/Training/FlatSurfaceLiquids//" #dataset folder
ListImages=os.listdir(os.path.join(TrainFolder)) # Create list of images
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    Img=cv2.imread(os.path.join(TrainFolder, ListImages[idx], "VesselWithContent_Frame_0_RGB.jpg")) # load image
    with open(os.path.join(TrainFolder, ListImages[idx],"ContentMaterial.json")) as f: # load liquid data
        MaterialPropeties = json.load(f)
    color=MaterialPropeties['Base Color'][:3] # read color
    # create tranasformation and transform image to pytorch
    transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    Img=transformImg(Img)
    return Img,torch.tensor(color)
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    color = torch.zeros([batchSize,3])
    for i in range(batchSize):
        images[i],color[i]=ReadRandomImage()
    return images,color
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.resnet50(pretrained=True) # Load net
Net.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True) # replace net final layer to  3 channels prediction
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------
AverageLoss=np.zeros([50])
for itr in range(500001): # Training loop
   images,prop=LoadBatch() # Load taining batch
   images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
   color = torch.autograd.Variable(prop, requires_grad=False).to(device) # Load annotation
   Pred = Net(images) # make prediction
   Net.zero_grad()
   Loss=torch.abs(Pred - color).mean()
   Loss.backward() # Backpropogate loss
   optimizer.step() # Apply gradient descent change to weight
   AverageLoss[itr%50]=Loss.data.cpu().numpy()
   print(itr,") Loss=",Loss.data.cpu().numpy(),'AverageLoss',AverageLoss.mean()) # Display loss
   if itr % 1000 == 0:
        print("Saving Model" +str(itr) + ".torch") #Save model weight
        torch.save(Net.state_dict(),   str(itr) + ".torch")
