import requests
import re
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def download_image(url):
    r = requests.get(url, allow_redirects=True)
    filename = get_filename_from_cd(r.headers.get('content-disposition'))
    if filename is None:
        if url.find('/'):
	        filename = url.rsplit('/', 1)[1]
        else:
            filename = 'inputfile.jpg'
    filename= 'data/downloads/'+filename
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


if __name__ == '__main__':
    import matplotlib
    import argparse
    import matplotlib.pyplot as plt

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

    args = parser.parse_args()
    args.device = torch.device("cpu")

    # setup model
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')

    if args.url is not None:
        image_file = download_image(args.url)

    if args.single_file is not None: 
        image_file = args.single_file
    print("running on ", image_file)

    # create name for files
    # mask_file = os.path.splitext(os.path.basename(image_file))[0] + '_mask.png'
    mask_file = 'mask.png'
    converted_file = os.path.splitext(os.path.basename(image_file))[0] + '_converted.png'

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(args.device)

    # Actually Apply the model
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)

    # only look at hair
    mask = mask >> 1
    mask=mask.squeeze()

    # save and reload it cause I don't know whats wrong
    matplotlib.image.imsave(mask_file,mask)
    # print("wrote mask to ", mask_file)
    mask = cv2.imread(mask_file)

    # create binary mask

    mask = cv2.resize(mask, dsize=image.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    grayImage = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    image_array = []
    # quick lut to change colors
    rgb_lut = [(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255)]
    # loop through all combinations of colors
    for rgb_val in rgb_lut:

        colorImg = np.zeros(image.shape, image.dtype)
        colorImg[:,:] = rgb_val
        colorMask = cv2.bitwise_and(colorImg, colorImg, mask=blackAndWhiteImage)
        weightedImage = cv2.addWeighted(colorMask, .5, image, 1, 0)
        completedImage = Image.fromarray(weightedImage)
        image_array.append(completedImage)


    # create gif with array of images 
    image_array[0].save('my.gif',save_all=True, append_images=image_array[1:], optimize=False, duration=400, loop=0)
    # converted_image = image


    # save to file
    # Image.fromarray(converted_image).save(converted_file)
    # matplotlib.image.imsave(converted_file,converted_image)

    print("wrote converted image to ", converted_file)

