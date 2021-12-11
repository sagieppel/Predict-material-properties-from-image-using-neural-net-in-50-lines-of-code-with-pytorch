import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import json

width=height=900 # image width and height
batchSize=4

TestImage="TEST1.jpg"
modelPath="10000.torch"
#---------------------Read image and transform to pytorch ---------------------------------------------------------

Img=cv2.imread(TestImage)
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
Img=transformImg(Img)

#--------------Load batch of images-----------------------------------------------------
images = torch.zeros([1,3,height,width])
images[0]=Img
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Select device for training
Net = torchvision.models.resnet50(pretrained=True) # Load net
Net.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True) # Replace final prediction to 3 values
Net = Net.to(device)
#----------------Train--------------------------------------------------------------------------
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set net to evaluation mode, usually usefull in this case its fail
#----------------Make preidction--------------------------------------------------------------------------
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0) # Convert to pytorch
with torch.no_grad():
    Prd = Net(Img)  # Run net
print("Predicte Liquid color RGB", Prd.data.cpu().numpy())
