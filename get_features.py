import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import argparse

# ------------------------------- STEP 3 --------------------------
# get features using a pretrained Resnet 18

# Use: -i data/rita -s resnet or -i optical_flow/rita -s optical_flow_features


class net(nn.Module):

    def __init__(self):
        super(net, self).__init__()
        model = models.resnet18(pretrained=True)
        if torch.cuda.is_available():
            self.net = model.cuda()
        else:
            self.net = model

        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, input):  # extract features from the average pooling layer

        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)  # [1, 512, 7, 7]
        output = self.net.avgpool(output)  # [1, 512, 1, 1]
        output = torch.flatten(output)  # flatten removes axis 1

        return output


def extractor_2d(path_to_frames, save_path):
    feature_extractor = net().to(device)
    feature_extractor.eval()  # to ensure that any Dropout layers are not active

    # For Resnet, the image must be at least 224, 224
    # transform image to (224x224) and normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    for img in sorted(os.listdir(path_to_frames)):  # list the images inside every subfolder

        img_path = os.path.join(path_to_frames, img)  # keep the path of every image
        print(img_path)
        img_name = os.path.splitext(img)[0]  # image name

        # remove .jpg extension and add .txt
        save_name = os.path.join(save_path, img_name + ".npy")  # 0001.npy

        img = Image.open(img_path)
        img = transform(img)

        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)  # [1, 3, 224, 224]
        x = x.cuda()

        y = feature_extractor(x)
        y = y.cpu().data.numpy()

        np.save(save_name, y)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', default="images", help='Path to images or image file')  # -i data/andy
    ap.add_argument('-s', '--save', default="images", help='Path to save directory')  # -s resnet
    args = ap.parse_args()

    data_dir = args.images
    output_dir = args.save

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    ############################################################################

    for folders in sorted(os.listdir(data_dir)):
        subfolder = os.path.join(data_dir, folders)  # [every subfolder] # data/andy/train

        for frames in os.listdir(subfolder):
            save_path = os.path.join(output_dir, os.path.split(data_dir)[-1], folders)

            #  create subfolders if they don't exist inside features folder
            if not os.path.isdir(save_path):
                os.mkdir(os.path.join(save_path))
                print("Created directory:  ", os.path.join(save_path))
            else:
                pass
            path_to_frames = os.path.join(subfolder)
            extractor_2d(path_to_frames, save_path)
