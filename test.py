from glob import glob
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

    args = parser.parse_args()
    args.device = torch.device("cpu")

    if args.single_file is None:
        image_files = sorted(glob('{}/*.jp*g'.format(args.data_folder)))
        model = load_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print('Model loaded')
        print(len(image_files), ' files in folder ', args.data_folder)

        # fig = plt.figure()
        for i, image_file in enumerate(image_files):
            if i >= args.batch_size:
                break

            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(image)
            torch_img = transform(pil_img)
            torch_img = torch_img.unsqueeze(0)
            torch_img = torch_img.to(args.device)

            # Forward Pass
            logits = model(torch_img)
            mask = np.argmax(logits.data.cpu().numpy(), axis=1)
            #img = Image.fromarray(mask.squeeze(), 'RGB')
            #img.save('outputmask.png')
            matplotlib.image.imsave('outputmask.png',mask.squeeze())
    else:
        # setup model
        model = load_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print('Model loaded')

        image_file = args.single_file
        print("running on ", image_file)

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image)
        torch_img = transform(pil_img)
        torch_img = torch_img.unsqueeze(0)
        torch_img = torch_img.to(args.device)

        # Actually Apply the model
        logits = model(torch_img)
        # print(logits.data.cpu().numpy())

        mask = np.argmax(logits.data.cpu().numpy(), axis=1)

        # only look at hair
        # mask = [[ 0 if val == 1 else val for val in line] for line in mask]
        # mask = [[ 1 if val == 2 else val for val in line] for line in mask]
        # mask = np.array(mask).unsqueeze()
        mask = mask >> 1
        # print(mask[0][10])
        # image[mask] = (0,0,255)

        # indices = np.where(mask==1)
        # image[indices[0], indices[1], :] = [0, 0, 255]

        mask = [[ 0 if val == 1 else val for val in line] for line in mask]

        converted_image = image

        # redImg = np.zeros(mask.shape, mask.dtype)
        # redImg[:,:] = (0, 0, 255)
        # redMask = cv2.bitwise_and(redImg, redImg, mask=mask)
        # cv2.addWeighted(redMask, 1, image, 1, 0, image)
        # # print(image)

        mask=mask.squeeze()

        # create name for mask file
        mask_file = os.path.splitext(os.path.basename(image_file))[0] + '_mask.png'
        converted_file = os.path.splitext(os.path.basename(image_file))[0] + '_converted.png'
        # save to file
        matplotlib.image.imsave(mask_file,mask)
        # Image.fromarray(converted_image).save(converted_file)
        matplotlib.image.imsave(converted_file,converted_image)

        print("wrote mask to ", mask_file)
        print("wrote converted image to ", converted_file)

