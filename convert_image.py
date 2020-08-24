import requests
import re
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
import matplotlib
import matplotlib.pyplot as plt

# load pre-trained model and weights
def load_model():

    model = MobileNetV2_unet(None).to(torch.device("cpu"))
    state_dict = torch.load('/project/checkpoints/model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def download_image(url):
    # download image from a url and return the filename
    r = requests.get(url, allow_redirects=True)
    # filename = get_filename_from_cd(r.headers.get('content-disposition'))
    # if filename is None:
    #     if url.find('/'):
	#         filename = url.rsplit('/', 1)[1]
    #     else:
    #         filename = 'inputfile.jpg'
    filename = 'input.png'
    filename= '/project/data/downloads/'+filename
    open(filename, 'wb').write(r.content)
    return filename

def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def image_to_rainbow_gif(image,binarymask,gif_file):
    # take single image and mask and output rainbow gif
    image_array = []
    # quick lut to change colors
    rgb_lut = [(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255)]
    # loop through all combinations of colors
    for rgb_val in rgb_lut:

        colorImg = np.zeros(image.shape, image.dtype)
        colorImg[:,:] = rgb_val
        colorMask = cv2.bitwise_and(colorImg, colorImg, mask=binarymask)
        weightedImage = cv2.addWeighted(colorMask, .5, image, 1, 0)
        completedImage = Image.fromarray(weightedImage)
        image_array.append(completedImage)

    # create gif with array of images 
    image_array[0].save(gif_file,save_all=True, append_images=image_array[1:], optimize=False, duration=400, loop=0)

def mask_to_binary_mask(mask, image):
    # only look at hair
    mask = mask >> 1
    mask=mask.squeeze()

    #this file is just temporary so its fine to just leave it
    mask_file = 'mask.png'

    # save and reload it cause I don't know whats wrong
    matplotlib.image.imsave(mask_file,mask)
    # print("wrote mask to ", mask_file)
    mask = cv2.imread(mask_file)

    # create binary mask
    mask = cv2.resize(mask, dsize=image.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return(blackAndWhiteImage)

def preprocess_image(transform, image_file):
    # read in and convert image
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #convert to a pytorch format
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))

    return(image, torch_img)


def main(args=None, url=None, single_file=None, output_folder=None):
    # setup model
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')

    # set up all the input args 
    # if args.url is not None:
    #     image_file = download_image(args.url)
    if url is not None:
        image_file = download_image(url)

    # if args.single_file is not None: 
    #     image_file = args.single_file
    if single_file is not None: 
        image_file = single_file

    # if args.output_folder is not None:
    #     output = str(args.output_folder) + '/output.gif'
    # else:
    #     output = 'output.gif'
    if output_folder is not None:
        output = str(output_folder) + '/output.gif'
    else:
        output = 'output.gif'

    print("running on ", image_file)

    # convert image to appropriate type
    (image,preprocessed_image) = preprocess_image(transform, image_file)

    # Actually Apply the model
    logits = model(preprocessed_image)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)

    # convert output mask to binary mask
    binarymask = mask_to_binary_mask(mask, image)

    # apply mask into rainbow
    image_to_rainbow_gif(image, binarymask, output)

    print("wrote gif to:", output)
    return (output)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default='./checkpoints/model.pt',
                        help='path of pre-trained weights (default: ./checkpoints/model.pt)')  

    parser.add_argument('-f','--single-file', type=str, default=None,
                        help='specify a single file to run on')

    parser.add_argument('-u','--url', type=str, default=None,
                        help='specify a url to download from')

    parser.add_argument('-o','--output_folder', type=str, default=None,
                        help='specify a path to write to')

    args = parser.parse_args()
    # args.device = torch.device("cpu")

    main(url=args.url, single_file=args.single_file, output_folder=args.output_folder)



