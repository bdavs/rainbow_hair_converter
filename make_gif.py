import animeface
import requests
from PIL import Image
import cv2
import numpy as np


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

    return (res2)

def noise_reduction(img):
    #noise reduction
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se2)

    return(opened)

def color_mask(color,img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(cvim, cv2.COLOR_BGR2HSV)

    #grab hair color
    color2 = color_object_to_list(color)
    # print(color2) #84,80,83
    haircolor = cv2.cvtColor(np.uint8([[color2]]),cv2.COLOR_BGR2HSV)
    # print(haircolor[0,0]) #143,12,84

    #hair hsv thresholds
    lower_color = np.array([haircolor[0,0,0]-25,0,50])
    # lower_red = np.array([30,150,50])
    upper_color = np.array([haircolor[0,0,0]+25,155,205])
    # upper_red = np.array([255,255,180])

    # get everything between the thresholds
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return(mask)

def contouring(original,mask):
    mask_new = np.zeros(mask.shape,np.uint8)
    contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 180<cv2.contourArea(cnt):
            cv2.drawContours(original,[cnt],0,(0,255,0),2)
            cv2.drawContours(mask_new,[cnt],0,255,-1)

    return(original,mask_new)

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
    # img = cvim.copy()

    faces = anime_face_features(image_file)

    # cluster the colors
    clustered = clustering(cvim)

    #create a mask from the colors
    mask = color_mask(faces[0].hair.color, clustered)

    #reduce the noise in the image
    mask = noise_reduction(mask)

    
    original,mask = contouring(original,mask)


    #show non masked area
    res = cv2.bitwise_and(cvim,cvim, mask= mask)

    #write everything to files
    cv2.imwrite('cvim.png',cvim)
    cv2.imwrite('mask.png',mask)
    cv2.imwrite('res.png',res)
    cv2.imwrite('contoured.png',original)
    cv2.imwrite('clustered.png',clustered)

    cvim_rgb = cv2.cvtColor(cvim, cv2.COLOR_BGR2RGB)
    image_to_rainbow_gif(cvim_rgb,mask,output)

if __name__ == '__main__':
    filename = '/files/Documents/GitHub/face-seg/data/downloads/input.png'
    main(single_file=filename,output_folder=".")

# rgb_val = faces[0].hair.color
# rgb_val = color2
# colorImg = np.zeros(cvim.shape, cvim.dtype)
# colorImg[:,:] = rgb_val
# cv2.imwrite("color.png", colorImg)
