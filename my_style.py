# encoding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision

from PIL import Image
import copy

num_epochs = 100
content_weight = 1
style_weight = 1000
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

loader = torchvision.transforms.ToTensor() # Transformed into torch tensors, their values are between 0 and 1.
unloader = torchvision.transforms.ToPILImage() 

def image_loader(image_name):
    img = Image.open(image_name)
    img = loader(img) 
    img = Variable(img).unsqueeze(0)
    img = img.type(dtype)

    return img

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.output = inputs.clone()
        if inputs.size() == self.target.size():
            self.loss = self.criterion(inputs * self.weight, self.target)
        return self.output

    def my_backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class GramMatrix(nn.Module):
    def forward(self, inputs):
        b, c, h, w = inputs.size()
        features = inputs.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.output = inputs.clone()
        self.G = self.gram(inputs)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def my_backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def get_model_and_losses(cnn, content_img, style_img):
    vgg19 = copy.deepcopy(cnn)

    content_losses = []
    style_losses = []

    transfer_model = nn.Sequential()
    gram = GramMatrix()

    if use_cuda:
        model = transfer_model.cuda()
        gram = gram.cuda()

    i = 1
    j = 1
    c_counter = 1
    s_counter = 1

    for layer in list(vgg19):
        if isinstance(layer, nn.Conv2d):
            name = "conv" + str(i) + "_" + str(j)
            transfer_model.add_module(name, layer)

            if name in content_layers:
                target = transfer_model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                transfer_model.add_module("content_loss_" + str(c_counter), content_loss)
                content_losses.append(content_loss)
                c_counter += 1

            if name in style_layers:
                target = transfer_model(style_img).clone()
                features_gram = gram(target)
                style_loss = StyleLoss(features_gram, style_weight)
                transfer_model.add_module("style_loss_" + str(s_counter), style_loss)
                style_losses.append(style_loss)
                s_counter += 1

        if isinstance(layer, nn.ReLU):
            name = "relu" + str(i) + "_" + str(j)
            transfer_model.add_module(name, layer)

            if name in content_layers:
                target = transfer_model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                transfer_model.add_module("content_loss_" + str(c_counter), content_loss)
                content_losses.append(content_loss)
                c_counter += 1

            if name in style_layers:
                target = transfer_model(style_img).clone()
                features_gram = gram(target)
                style_loss = StyleLoss(features_gram, style_weight)
                transfer_model.add_module("style_loss_" + str(s_counter), style_loss)
                style_losses.append(style_loss)
                s_counter += 1

            j += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool" + str(i)
            transfer_model.add_module(name, layer)

            # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            # transfer_model.add_module(name, avgpool)

            i += 1
            j = 1

    return transfer_model, content_losses, style_losses


def transfer(content_image_name, style_image_name, result_image_name):
    print("Loading vgg model...")
    content_img = image_loader(content_image_name)
    style_img = image_loader(style_image_name)
    input_param = image_loader(content_image_name)
    input_param = nn.Parameter(input_param.data) # 默认requires_grad=True
    optimizer = optim.LBFGS([input_param])
    cnn = torchvision.models.vgg19(pretrained=True).features
    if use_cuda:
        cnn = cnn.cuda()

    print("Constructing transfer_model...")
    transfer_model, content_losses, style_losses = get_model_and_losses(cnn, content_img, style_img)

    print("Generating picture:")
    step = [0]
    while step[0] < num_epochs:
        def closure():
            input_param.data.clamp_(0, 1)
            optimizer.zero_grad()
            transfer_model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.my_backward()
            for cl in content_losses:
                content_score += cl.my_backward()

            step[0] += 1
            if step[0] == 1 or step[0] % 50 == 0:
                print("step {}:".format(step))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    result = input_param.squeeze(0)
    torchvision.utils.save_image(result.data, result_image_name)
    print("Done!")

if __name__ == "__main__":
    transfer("images/content.png","images/style.png", "images/result.png")


