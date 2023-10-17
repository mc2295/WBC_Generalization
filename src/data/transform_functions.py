import torch
from fastai.vision.all import PILImage, Tensor, Transform, Resize
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
image_size = 226
def transform_color_transfer(img: PILImage):
    sigma_targ = torch.tensor([0.45889156503010636, 0.15995689328283794, 0.02254376531226124])
    average_targ = torch.tensor([-0.4052762362389589, -0.07202646968215255, 0.01747981861569773])
    average1, sigma1 = average_targ.reshape(-1,1,1), sigma_targ.reshape(-1,1,1)

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

    return img_transformed.float()

def transform_resolution(img : PILImage):
    new_size = random.randint(50, 200)
    img2 = img.resize((new_size,new_size))
    img2 = img2.resize((image_size,image_size))
    return img2

def transform_crop(img: PILImage):

    # n = random.randint(100,200)
    # n = 200
    new_width, new_height = 300,200
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    img2 = img.crop((left, top, right, bottom))
    img2 = img2.resize((image_size,image_size))
    return img2

def get_item(row, entry_path) -> PILImage:
    image_path = row['name']  # Assuming 'image_path' column contains the file paths of the images
    label = row['label']  # Assuming 'label' column contains the labels for the images
    # image = open_image('../', image_path)
    image = PILImage.create(entry_path+ image_path)
    dataset = row['dataset']
    if dataset in ['rabin', 'matek', 'vexas', 'matek_2018', 'ruinjing', 'saint_antoine', 'lisc', 'jslh', 'jin_woo_choi', 'jiangxi_tecom', 'cella_vision_blog', 'tianjin', 'barcelona', 'vexas_original']:
        image = transforms.Compose([Transform(transform_color_transfer), transforms.Resize(image_size)])(image) # usually is 224
        image = (image.permute(2,1,0).numpy()*255).clip(0, 255).astype(np.uint8)
        pil_image = PILImage.create(image)
        return pil_image
    # elif dataset in ['barcelona']:
    #     image = transforms.Compose([Transform(transform_resolution), Transform(transform_crop), Transform(transform_color_transfer), transforms.Resize(image_size)])(image)
    #     image = (image.permute(2,1,0).numpy()*255).clip(0, 255).astype(np.uint8)
    #     pil_image = PILImage.create(image)
        return pil_image
    elif dataset in ['bccd']:
        image = transforms.Compose([Transform(transform_crop), Transform(transform_color_transfer), transforms.Resize(image_size)])(image)
        image = (image.permute(2,1,0).numpy()*255).clip(0, 255).astype(np.uint8)
        pil_image = PILImage.create(image)
        return pil_image
