import animeface
import requests
import math
import os
from PIL import Image
import cv2
import numpy as np
import default_parameters as dp 
import gif_resize

def bound(minx, x, maxx):
    if x < minx: return minx
    if x > maxx: return maxx
    return x

def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def color_object_to_list(color):
    # rgb = []
    # rgb.append(color.r)
    # rgb.append(color.g)
    # rgb.append(color.b)

    bgr = []
    bgr.append(color.b)
    bgr.append(color.g)
    bgr.append(color.r)
    return(bgr)

def download_image(url):
    # download image from a url and return the filename
    r = requests.get(url, allow_redirects=True)
    filename = 'input.png'
    filename= '/project/data/downloads/'+filename
    open(filename, 'wb').write(r.content)
    return filename

def anime_face_features(filename):
    #animefaces detection
    im = Image.open(filename)
    faces = animeface.detect(im)
    # print(faces)
    return(faces)

def clustering(img):
    #clustering
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res1 = center[label.flatten()]
    res2 = res1.reshape((img.shape))

    if dp.dev:
        cv2.imwrite('alt_methods/clustered.png',clustered)

    return (res2)

def color_mask(color,img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(cvim, cv2.COLOR_BGR2HSV)

    # print(color2) #84,80,83
    haircolor =   cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)
    # print(haircolor[0,0]) #143,12,84

    #hair hsv thresholds
    hrange = dp.hrange
    srange = dp.srange
    vrange = dp.vrange
    lower_color = np.array([bound(0,haircolor[0,0,0]-hrange,180),
                            bound(0,haircolor[0,0,1]-srange,255),
                            bound(0,haircolor[0,0,2]-vrange,255)
                            ])
    # lower_red = np.array([30,150,50])
    upper_color = np.array([bound(0,haircolor[0,0,0]+hrange,180),
                            bound(0,haircolor[0,0,1]+srange,255),
                            bound(0,haircolor[0,0,2]+vrange,255)
                            ])
    # upper_red = np.array([255,255,180])

    # get everything between the thresholds
    mask = cv2.inRange(hsv, lower_color, upper_color)

    if dp.dev:
        cv2.imwrite('alt_methods/initmask.png',mask)

    return(mask)

def make_gradient(img,style=cv2.COLORMAP_HSV):
        # create a gradient
    white_image = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    white_image.fill(255)

    rampl = np.linspace(1, 0, img.shape[0])
    rampl = np.tile(np.transpose(rampl), (img.shape[1],1))
    rampl = np.transpose(rampl)
    gradient = cv2.merge([rampl,rampl,rampl])
    # gradient = np.repeat(np.tile((img.shape[1], 1),np.linspace(1, 0, img.shape[0]))[:, :, np.newaxis], 3, axis=2)
    # gradient = np.repeat(np.tile(np.linspace(1, 0, img.shape[1]), (img.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
    gradient = gradient * white_image
    gradient = np.uint8(gradient)
    gradient = cv2.applyColorMap(gradient, style)
    
    if dp.dev:
        cv2.imwrite('alt_methods/gradient.png',gradient)

    return(gradient)

def make_gradient_array(img,num_colors):
    width = 180//num_colors

    gradient = make_gradient(img)
    grad_array = []
    h_mov=0
    for i in range(num_colors):

        hsv_grad = cv2.cvtColor(gradient,cv2.COLOR_RGB2HSV)
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_val, s, v = cv2.split(hsv_grad)
        # ((h.astype('int16') + shift_h) % 180).astype('uint8')
        shift_h = ((h_val.astype('int16') + h_mov) % 180).astype('uint8')
        shift_hsv = cv2.merge([shift_h, s, v])
        shift_img = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2RGB)

        grad_array.append(shift_img)
        h_mov += width

        if dp.dev:
            # print("wrote image")
            cv2.imwrite('alt_methods/colors/color_mask{}.png'.format(i),shift_img)

    print(len(grad_array))
    return(grad_array)

def noise_reduction(img):
    #noise reduction
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # img = cv2.erode(img, se1)

    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1,iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se2,iterations=3)

    # img = cv2.dilate(img, se2,iterations=3)

    if dp.dev:
        cv2.imwrite('alt_methods/noise_reduced.png',opened)

    return(opened)

def contouring(original,mask,min_island_size=dp.min_island_size):
    mask_new = np.zeros(mask.shape,np.uint8)
    contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if min_island_size<cv2.contourArea(cnt):
            cv2.drawContours(original,[cnt],0,(0,255,0),2)
            cv2.drawContours(mask_new,[cnt],0,255,-1)

    if dp.dev:
        cv2.imwrite('alt_methods/contoured.png',original)

    return(mask_new)

def image_to_rainbow_gif(image,binarymask,gif_file,num_colors=dp.num_colors):
    # take single image and mask and output rainbow gif
    image_array = []

    # num_colors = 5
    rgb_lut = []

    h=0
    s=dp.saturation
    v=dp.valueC
    width = 180//num_colors
    for i in range(num_colors):
        rgb_lut.append(cv2.cvtColor(np.uint8([[[h,s,v]]]),cv2.COLOR_HSV2RGB)[0,0].tolist())
        h += width

    if dp.gradient is True:
        grad_array = make_gradient_array(image,num_colors)

    # prepare the mask
    prepared_mask = np.zeros(image.shape, image.dtype)
    prepared_mask[:,:,0] = binarymask
    prepared_mask[:,:,1] = binarymask
    prepared_mask[:,:,2] = binarymask
    zeros = np.zeros(image.shape, image.dtype)
    prepared_mask = prepared_mask / 255

    # i = 0
    # loop through all combinations of colors
    for i in range(num_colors):
    # for rgb_val in rgb_lut:

        colorImg = np.zeros(image.shape, image.dtype)
        if dp.gradient is True:
            colorImg[:,:] = grad_array[i]
        else:
            colorImg[:,:] = rgb_lut[i]
        # colorImg[:,:] = grad_array[i]
        # rgb_val
        # i+= width
        color_mask = cv2.bitwise_and(colorImg, colorImg, mask=binarymask) 

        
        # apply mask to color       
        color_mask = np.uint8(colorImg * prepared_mask + zeros * (1 - prepared_mask))
        weightedImage = cv2.addWeighted(color_mask, dp.mask_opacity, image, 1, 0)

        # completedImage = Image.fromarray(colorImg)
        completedImage = Image.fromarray(weightedImage)
        
        # completedImage = completedImage.resize((600,800),Image.ANTIALIAS)
        image_array.append(completedImage)

    # create gif with array of images 
    
    image_array[0].save(gif_file,save_all=True, append_images=image_array[1:], optimize=True, duration=500, loop=0)
    filesize = os.path.getsize(gif_file)
    print(filesize)
    
    # resize large gif to allow them to be sent over discord
    while filesize > 8000000 and dp.size_check is True:
        gif = Image.open(gif_file)
        resize_to = (gif.size[0] //1.33, gif.size[1] // 1.33)
        gif.close()
        gif_resize.resize_gif(gif_file,resize_to=resize_to)
        filesize = os.path.getsize(gif_file)
        print(filesize)


def main(url=None, single_file=None, output_folder=None):

    # set up all the input args 
    if url is not None:
        image_file = download_image(url)

    if single_file is not None: 
        image_file = single_file
        
    if output_folder is not None:
        output = str(output_folder) + '/output.gif'
    else:
        output = '/project/data/output/output.gif'

    #open the file
    cvim = cv2.imread(image_file)
    original = cvim.copy()

    #get facial feautures, namely hair color
    faces = anime_face_features(image_file)

    #no faces identified, time to guess
    if not faces:
        hair_color = cvim[cvim.shape[0]//4,cvim.shape[1]//2]
    else:
        #grab hair color
        hair_color = color_object_to_list(faces[0].hair.color)

    #create a mask from the hair colors
    clusterLevel = dp.clusterLevel
    if clusterLevel > 0:
        # cluster the colors
        clustered = clustering(cvim)
        mask = color_mask(hair_color, clustered)
    else:
        mask = color_mask(hair_color, cvim)


    # gradient = make_gradient(cvim)


    #reduce the noise in the image
    mask = noise_reduction(mask)

    #trace around the largest areas
    mask = contouring(original,mask)

    # add blur to feather/smooth edges
    mask = cv2.GaussianBlur(mask,(21,21),5)
    # _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # mask = cv2.GaussianBlur(mask,(5,5),5)
    # mask = cv2.blur(mask,(5,5))



    # dilate for testing
    # se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.dilate(mask, se2,iterations=1)



    #write everything to files for development
    if dp.dev:
        #show non masked area
        res = cv2.bitwise_or(cvim,cvim, mask= mask)
        cv2.imwrite('alt_methods/cvim.png',cvim)
        # cv2.imwrite('alt_methods/gradient.png',gradient)
        cv2.imwrite('alt_methods/mask.png',mask)
        cv2.imwrite('alt_methods/res.png',res)

    # convert to rgb
    cvim_rgb = cv2.cvtColor(cvim, cv2.COLOR_BGR2RGB)

    #execute rainbow function
    image_to_rainbow_gif(cvim_rgb,mask,output)


if __name__ == '__main__':
    filename = '/project/data/downloads/input.png'
    main(single_file=filename,output_folder=".")
