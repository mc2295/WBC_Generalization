import torch
from fastai.vision.all import PILImage, Tensor
from torchvision import transforms
import numpy as np
import random

'''
different transforms :
- colour balance based on Gray average
- colour_balance_on_white needs a white zone and take it as white reference
- transform_color_transfer : color transfer by putting everything under the same mean and std in alphabetal space
- transform_resolution: randomly degrades resolution
- transform_crop: randomly crops image (centered crop)
'''

def colour_balance(img: PILImage):
    img = torch.tensor(np.array(img.resize((224,224)))).permute(2,0,1).float()/255
    R_mean, G_mean ,B_mean = torch.mean(img, dim = [1, 2])
    img_grey = transforms.Grayscale()(img)
    mean_grey = torch.mean(img_grey, dim = [1, 2])
    mean_color = [R_mean, G_mean ,B_mean]
    weight = torch.tensor(np.ones((3, 224,224)))
    for i in range(3):
        weight[i,:,:] = weight[i,:,:]*mean_grey/mean_color[i]
    new_img = torch.mul(img, torch.tensor(weight))
    return new_img.float()

def colour_balance_on_white(img: PILImage):
    from_row, from_column, row_width, column_width = 10, 10,200,200
    img = torch.tensor(np.array(img.resize((224,224)))).permute(2,0,1).float()/255
    img = img[:3,:,:]
    # image_patch = img[:,from_row:from_row+row_width,
    #                     from_column:from_column+column_width]
    image_patch = img.clone().detach()
    # image_patch = img ## to visualise the black square
    image_patch[:,from_row:from_row+row_width,
                        from_column:from_column+column_width] = torch.tensor(np.zeros_like(image_patch[:,from_row:from_row+row_width,
                        from_column:from_column+column_width]))
    rgb1 =1./ image_patch.flatten(-2).max(dim=1).values
    rgb1 = rgb1.reshape(-1, 1, 1)
    new_img = rgb1 * img
    return new_img

def rgb2lalphabeta_star(img: PILImage):
    img = torch.tensor(np.array(img.resize((224,224)))).permute(2,1,0).float()/255
    img = img[:3,:,:]
    mitu = torch.tensor([[0.3811, 0.5783, 0.0402],[0.1967, 0.7244, 0.0782], [0.0241,0.1288,0.8444]])
    mitu = mitu.reshape(3,3,1,1)

    lms = torch.log(torch.sum(mitu*img, dim = 1))

    m = torch.sum(torch.tensor([[1,1,1], [1,1,-2], [1,-1,0]]).reshape(3,3,1,1)*lms, dim=1)

    lalphabeta = torch.sum(torch.tensor([[1/np.sqrt(3),0,0], [0,1/np.sqrt(6),0], [0,0, 1/np.sqrt(2)]]).reshape(3,3,1,1)*m, dim=1)

    lalphabeta_star = lalphabeta - torch.mean(lalphabeta, dim = [1,2]).reshape(-1,1,1)
    return lalphabeta_star, torch.mean(lalphabeta, dim = [1,2]).reshape(-1,1,1)

def lalphabeta2rgb(lalphabeta: Tensor):
    m = torch.sum(torch.tensor([[np.sqrt(3)/3,0,0], [0,np.sqrt(6)/6,0], [0,0, np.sqrt(2)/2]]).reshape(3,3,1,1)*lalphabeta, dim=1)
    lms_log = torch.sum(torch.tensor([[1,1,1], [1,1,-1], [1,-2,0]]).reshape(3,3,1,1)*m, dim=1)
    lms = torch.exp(lms_log)
    img_rgb = torch.sum(torch.tensor([[4.4679, -3.5873,0.1193], [-1.2186,2.3809,-0.1624], [0.0497,-0.2439,1.2045]]).reshape(3,3,1,1)*lms, dim=1)
    return img_rgb

def transform_color_transfer(img: PILImage):
    sigma_saint_antoine = torch.tensor([0.45889156503010636, 0.15995689328283794, 0.02254376531226124])
    average_saint_antoine = torch.tensor([-0.4052762362389589, -0.07202646968215255, 0.01747981861569773])

    sigma_barcelona = torch.tensor([0.44521334705391624, 0.16415471833786074, 0.01225855119375951])
    average_barcelona = torch.tensor([-0.5305400947047252, 0.03253191439619589, 0.024497598523987463])

    average1, sigma1 = average_saint_antoine.reshape(-1,1,1), sigma_saint_antoine.reshape(-1,1,1)

    img = np.array(img.resize((224,224)))
    img[np.where(img == 0)] = 1
    img = torch.tensor(img).permute(2,1,0).float()/255
    img = img[:3,:,:]
    mitu = torch.tensor([[0.3811, 0.5783, 0.0402],[0.1967, 0.7244, 0.0782], [0.0241,0.1288,0.8444]])
    mitu = mitu.reshape(3,3,1,1)
    lms = torch.log(torch.sum(mitu*img, dim = 1))
    m = torch.sum(torch.tensor([[1,1,1], [1,1,-2], [1,-1,0]]).reshape(3,3,1,1)*lms, dim=1)
    lalphabeta = torch.sum(torch.tensor([[1/np.sqrt(3),0,0], [0,1/np.sqrt(6),0], [0,0, 1/np.sqrt(2)]]).reshape(3,3,1,1)*m, dim=1)
    lalphabeta = lalphabeta - torch.mean(lalphabeta, dim = [1,2]).reshape(-1,1,1)
    sigma2 = torch.std(lalphabeta, dim = [1,2]).reshape(-1,1,1)
    lalphabeta = sigma1/sigma2*lalphabeta + average1
    m = torch.sum(torch.tensor([[np.sqrt(3)/3,0,0], [0,np.sqrt(6)/6,0], [0,0, np.sqrt(2)/2]]).reshape(3,3,1,1)*lalphabeta, dim=1)
    lms_log = torch.sum(torch.tensor([[1,1,1], [1,1,-1], [1,-2,0]]).reshape(3,3,1,1)*m, dim=1)
    lms = torch.exp(lms_log)
    img_transformed = torch.sum(torch.tensor([[4.4679, -3.5873,0.1193], [-1.2186,2.3809,-0.1624], [0.0497,-0.2439,1.2045]]).reshape(3,3,1,1)*lms, dim=1)


    # lalphabeta_star2, average2 = rgb2lalphabeta_star(img_to_transform)
    # sigma1 = sigma1.reshape(-1,1,1)
    # average1 = average1.reshape(-1,1,1)
    # sigma2 = torch.std(lalphabeta_star2, dim = [1,2]).reshape(-1,1,1)
    # lalphabeta_star2 = sigma1/sigma2*lalphabeta_star2 + average1
    # img_transformed = lalphabeta2rgb(lalphabeta_star2)
    return img_transformed.float()

def transform_resolution(img : PILImage):
    new_size = random.randint(50, 200)
    img2 = img.resize((new_size,new_size))
    img2 = img2.resize((224,224))
    return img2

def transform_crop(img: PILImage):
    n = random.randint(100,200)
    new_width, new_height = n,n
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img2 = img.crop((left, top, right, bottom))
    img2 = img2.resize((224,224))
    return img2
