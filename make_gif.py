import animeface
import requests
import math
import os
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import default_parameters as dp 
import gif_resize
import ffmpeg


# dp.MAX_DISCORD_FILE_SIZE = 8000000

def bound(minx, x, maxx):
    if x < minx: return minx
    if x > maxx: return maxx
    return x

def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def color_object_to_list(color):
    # convert rgb tuple to list
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

def anime_face_features(filename=None,input_stream=None):
    #animefaces detection
    if filename is not None:
        im = Image.open(filename)
    elif input_stream is not None:
        im = Image.open(input_stream)
    else:
        raise
    faces = animeface.detect(im)
    # print(faces)
    return(faces)

def clustering(img,clusterLevel):
    #clustering
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    
    #define number of clusters
    if clusterLevel == 1:
        K = 8
    elif clusterLevel == 2:
        K = 7
    elif clusterLevel == 3:
        K = 6
    elif clusterLevel >= 4:
        K = 5
    else: 
        K = 6
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res1 = center[label.flatten()]
    res2 = res1.reshape((img.shape))

    if dp.dev:
        cv2.imwrite('alt_methods/clustered.png',res2)

    return (res2)

def color_mask_range(options):
    # change ranges based on fill level
    if options is not None:
        if options['fill_level'] <= 0:
            hrange = 5
            srange = 10
            vrange = 10
        elif options['fill_level'] == 1:
            hrange = 10
            srange = 20
            vrange = 20
        elif options['fill_level'] == 2:
            hrange = 15
            srange = 30
            vrange = 30
        elif options['fill_level'] == 3:
            hrange = 17
            srange = 40
            vrange = 40
        elif options['fill_level'] == 4:
            # defaults
            hrange = 20
            srange = 50
            vrange = 50
        elif options['fill_level'] == 5:
            hrange = 25
            srange = 60
            vrange = 60
        elif options['fill_level'] == 6:
            hrange = 30
            srange = 70
            vrange = 70
        elif options['fill_level'] == 7:
            hrange = 40
            srange = 80
            vrange = 80
        elif options['fill_level'] == 8:
            hrange = 55
            srange = 90
            vrange = 90
        elif options['fill_level'] >= 9:
            hrange = 80
            srange = 127
            vrange = 127
        else :
            hrange = dp.hrange
            srange = dp.srange
            vrange = dp.vrange
    else:
        hrange = dp.hrange
        srange = dp.srange
        vrange = dp.vrange

    return(hrange,srange,vrange)


def color_mask(color,img,options):
    # convert the image into an hsv image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #convert hair color to hsv color
    haircolor =   cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)

    #get hsv ranges from options input
    hrange,srange,vrange = color_mask_range(options)

    #hair hsv thresholds
    lower_color = np.array([bound(0,haircolor[0,0,0]-hrange,179),
                            bound(0,haircolor[0,0,1]-srange,255),
                            bound(0,haircolor[0,0,2]-vrange,255)
                            ])
    upper_color = np.array([bound(0,haircolor[0,0,0]+hrange,179),
                            bound(0,haircolor[0,0,1]+srange,255),
                            bound(0,haircolor[0,0,2]+vrange,255)
                            ])


    # get everything between the thresholds
    mask = cv2.inRange(hsv, lower_color, upper_color)

    if dp.dev:
        cv2.imwrite('alt_methods/initmask.png',mask)

    return(mask)


def noise_reduction(img):
    #noise reduction
    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    # img = cv2.erode(img, se1)
    img = cv2.dilate(img, se1,iterations=1)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1,iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se2,iterations=3)

    if dp.dev:
        cv2.imwrite('alt_methods/noise_reduced.png',opened)

    return(opened)

def contouring(img,mask,min_island_size=dp.min_island_size):

    min_island_size = min(mask.shape[0],mask.shape[1]) * 2.5
    mask_new = np.zeros(mask.shape,np.uint8)
    contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if min_island_size<cv2.contourArea(cnt):
            cv2.drawContours(mask_new,[cnt],0,255,-1)

    if dp.dev:
        cv2.imwrite('alt_methods/mask_new.png',mask_new)

    return(mask_new)

def make_gradient(img,style=cv2.COLORMAP_HSV):
    # create a gradient
    white_image = np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
    white_image.fill(255)

    rampl = np.linspace(1, 0, img.shape[0])
    rampl = np.tile(np.transpose(rampl), (img.shape[1],1))
    rampl = np.transpose(rampl)
    gradient = cv2.merge([rampl,rampl,rampl])
    gradient = gradient * white_image
    gradient = np.uint8(gradient)
    gradient = cv2.applyColorMap(gradient, style)

    return(gradient)

def make_gradient_array(img,num_colors):
    width = 180//num_colors
    
    h_mov=0
    grad_array = []

    gradient = make_gradient(img)
    hsv_grad = cv2.cvtColor(gradient,cv2.COLOR_RGB2HSV)
    h_val, s, v = cv2.split(hsv_grad)
    
    for i in range(num_colors):    
        # bound the h value and temporarily convert to int16
        shift_h = ((h_val.astype('int16') + h_mov) % 180).astype('uint8')
        shift_hsv = cv2.merge([shift_h, s, v])
        shift_img = cv2.cvtColor(shift_hsv, cv2.COLOR_HSV2RGB)

        grad_array.append(shift_img)
        h_mov += width


    return(grad_array)

def image_to_rainbow_gif(image,binarymask,gif_file,num_colors=dp.num_colors,options=None):
    # take single image and mask and output rainbow gif
    image_array = np.ndarray()

    if options is not None:
        if options['gradient'] >= 1:
            use_gradient = True
        else:
            use_gradient = False
        if options['num_colors']:
            num_colors = options['num_colors']
    else:
        use_gradient = dp.gradient

    # num_colors = 5
    rgb_lut = []

    h=dp.hueStart
    s=dp.saturation
    v=dp.valueC
    width = 180//num_colors
    for i in range(num_colors):
        rgb_lut.append(cv2.cvtColor(np.uint8([[[h,s,v]]]),cv2.COLOR_HSV2RGB)[0,0].tolist())
        h += width

    if use_gradient is True:
        grad_array = make_gradient_array(image,num_colors)

    # prepare the mask
    prepared_mask = np.zeros(image.shape, image.dtype)
    prepared_mask[:,:,0] = binarymask
    prepared_mask[:,:,1] = binarymask
    prepared_mask[:,:,2] = binarymask
    zeros = np.zeros(image.shape, image.dtype)
    prepared_mask = prepared_mask / 255

    # loop through all combinations of colors
    for i in range(num_colors):

        colorImg = np.zeros(image.shape, image.dtype)
        if use_gradient is True:
            colorImg[:,:] = grad_array[i]
        else:
            colorImg[:,:] = rgb_lut[i]

        color_mask = cv2.bitwise_and(colorImg, colorImg, mask=binarymask) 

        # apply mask to color       
        color_mask = np.uint8(colorImg * prepared_mask + zeros * (1 - prepared_mask))
        weightedImage = cv2.addWeighted(color_mask, dp.mask_opacity, image, dp.original_opacity, 0)

        completedImage = Image.fromarray(weightedImage)
        
        # convert back to rgb
        completedImage = cv2.cvtColor(np.array(completedImage), cv2.COLOR_BGR2RGB)
        image_array.append(completedImage)


    animated_gif = BytesIO()
    image_array[0].save(animated_gif,format='GIF', save_all=True, append_images=image_array[1:], optimize=True, duration=500, loop=0)
 
    animated_gif.seek(0,2)
    filesize = animated_gif.tell()

    animated_gif.seek(0)

    # # create video file
    # output_vid = '/project/data/output/output2.mp4' 
    # height, width, layers = image.shape
    # video = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*'mp4v'), 6, (width,height))

    # # add images to video
    # for image_a in range(len(image_array)):
    #     video.write(image_array[image_a])

    # # clean up
    # cv2.destroyAllWindows()
    # video.release()

    # output_size = os.path.getsize(output_vid)//1000


    # print ('file image size kb= ', output_size)

    # output_buf.seek(0,2)
    # output_size = output_buf.tell()//1000
    # print ('byteio image size kb= ', output_size)
    # output_buf.seek(0) 

    # max image upload for bots is 8GB
    # target_size = 8000
    # if output_size > target_size:
    # # image too large for discord
    #     new_output_vid = '/project/data/output/output3.mp4' 

    #     compress_video(output_vid,new_output_vid,target_size)

    #     # read from file to bytesio
    #     with open(new_output_vid, 'rb') as f:
    #         output_buf = BytesIO(f.read())

    # else:
    # # image size fine for discord
    #     # read from file to bytesio
    #     with open(output_vid, 'rb') as f:
    #         output_buf = BytesIO(f.read())

    # return(output_buf)

    # write to files
    # with open(gif_file, "wb") as f:
    #     f.write(output_vid.getbuffer())


    # resize large gif to allow them to be sent over discord
    # while filesize > dp.MAX_DISCORD_FILE_SIZE and dp.size_check is True:
    # #     print(f'current filesize is {filesize}, reducing')
    #     gif = Image.open(animated_gif)
    #     resize_to = (gif.size[0] //1.33, gif.size[1] // 1.33)
    #     # gif.close()
    #     gif_resize.resize_gif(animated_gif,resize_to=resize_to)
    #     animated_gif.seek(0,2)
    #     filesize = animated_gif.tell()
    #     print ('new GIF image size kb= ', filesize//1000)
    #     # Optional: write contents to file
    #     animated_gif.seek(0)
    #     filesize = os.path.getsize(gif_file)

    return(animated_gif)
    # print(f'final filesize {filesize}, saved')

def maskFromFilePair(image1, image2, margarineForError):
    img = Image.new('RGB', [image1.width,image1.height], 255)
    dataNew = img.load()
    data1=image1.load()
    data2=image2.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if abs(data1[x,y][0]-data2[x,y][0])<margarineForError \
            and abs(data1[x,y][1]-data2[x,y][1])<margarineForError \
            and abs(data1[x,y][2]-data2[x,y][2])<margarineForError:
                dataNew[x,y] = (0,0,0)
            else:
                dataNew[x,y] = (255,255,255)
    img = cv2.cvtColor(np.array(img), cv2.cv2.COLOR_RGB2GRAY)     
    return img

def get_image_size(img,imgtype:str=None):
    # cvim_save = Image.fromarray(img)
    onefile = BytesIO()
    img.save(onefile,format='GIF', optimize=True, duration=500, loop=0)
    onefile.seek(0,2)
    print (imgtype,'image size kb= ', onefile.tell()//1000)
    onefile.seek(0)   

def compress_video(video_full_path, output_file_name, target_size):
    # target size is in kb
    # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    min_audio_bitrate = 32000
    max_audio_bitrate = 256000

    probe = ffmpeg.probe(video_full_path)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    # # Audio bitrate, in bps.
    # audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
    audio_bitrate = float(min_audio_bitrate)
    # # Target total bitrate, in bps.
    target_total_bitrate = (target_size * 1024 * 8) / (1.073741824 * duration)

    # # Target audio bitrate, in bps
    # if 10 * audio_bitrate > target_total_bitrate:
    #     audio_bitrate = target_total_bitrate / 10
    #     if audio_bitrate < min_audio_bitrate < target_total_bitrate:
    #         audio_bitrate = min_audio_bitrate
    #     elif audio_bitrate > max_audio_bitrate:
    #         audio_bitrate = max_audio_bitrate
    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate 
    # - audio_bitrate

    i = ffmpeg.input(video_full_path)
    ffmpeg.output(i, '/dev/null' if os.path.exists('/dev/null') else 'NUL',
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                  ).overwrite_output() .global_args('-loglevel', 'error').run()
    ffmpeg.output(i, output_file_name,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
                  ).overwrite_output() .global_args('-loglevel', 'error').run()

def main_dual(input_stream_image=None,input_stream_mask=None,options=None):
    if input_stream_image is not None and input_stream_mask is not None:
        # get mask
        cvim_mask = maskFromFilePair(Image.open(input_stream_image),Image.open(input_stream_mask), 16)

        # image section
        input_stream_image.seek(0,2)
        image_size = input_stream_image.tell()
        input_stream_image.seek(0)
        cvim = cv2.imdecode(np.frombuffer(input_stream_image.getbuffer(), np.uint8), cv2.IMREAD_COLOR)

        # # mask section
        # input_stream_mask.seek(0,2)
        # mask_size = input_stream_mask.tell()
        # input_stream_mask.seek(0)
        # cvim_mask = cv2.imdecode(np.frombuffer(input_stream_mask.getbuffer(), np.uint8), cv2.IMREAD_COLOR)

        # sanity check
        if cvim.shape != cvim_mask.shape:
            # not the same size, quit
            return

        # # size management
        if cvim.shape[1] > 2000:
            resize_factor = float(cvim.shape[1] / 2000)
            cvim = cv2.resize(cvim, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
            cvim_mask = cv2.resize(cvim_mask, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
        if cvim.shape[0] > 2000:
            resize_factor = float(cvim.shape[0] / 2000)
            cvim = cv2.resize(cvim, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
            cvim_mask = cv2.resize(cvim_mask, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
    else:
        return


    # convert to rgb
    cvim_rgb = cv2.cvtColor(cvim, cv2.COLOR_BGR2RGB)

    output = '/project/data/output/output.gif'
    
    #execute rainbow function
    animated_gif = image_to_rainbow_gif(cvim_rgb,cvim_mask,output,options=options)

    return(animated_gif)

def main(url=None, single_file=None, output_folder=None, input_stream=None, options=None):

    # set up all the input args 
    if url is not None:
        image_file = download_image(url)

    if single_file is not None: 
        image_file = single_file
        
    if output_folder is not None:
        output = str(output_folder) + '/output.gif'
    else:
        output = '/project/data/output/output.gif'

    if options is not None:
        # cluster if fill level larger than 6
        clusterLevel = max(options['fill_level']-6,0)
    else:
        clusterLevel = dp.clusterLevel
        # clusterLevel = 0

    if input_stream is not None:
        input_stream.seek(0,2)
        image_size = input_stream.tell()
        input_stream.seek(0)
        cvim = cv2.imdecode(np.frombuffer(input_stream.getbuffer(), np.uint8), cv2.IMREAD_COLOR)


        # # size management
        if cvim.shape[1] > 2000:
            resize_factor = float(cvim.shape[1] / 2000)
            cvim = cv2.resize(cvim, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
        elif cvim.shape[0] > 2000:
            resize_factor = float(cvim.shape[0] / 2000)
            cvim = cv2.resize(cvim, (int(cvim.shape[1]//resize_factor),int(cvim.shape[0]//resize_factor)))
            

        # predicted_image_size = image_size * 5 * int(options['num_colors'])
        # if predicted_image_size > dp.MAX_DISCORD_FILE_SIZE:
        #     estimated_resize = predicted_image_size/dp.MAX_DISCORD_FILE_SIZE
        #     normalized_resize = math.sqrt(estimated_resize)
        #     cvim = cv2.resize(cvim, (int(cvim.shape[1]//normalized_resize),int(cvim.shape[0]//normalized_resize)))
            
        #get facial feautures, namely hair color
        faces = anime_face_features(input_stream=input_stream)

    #open the file (deprecate this)
    else:
        cvim = cv2.imread(image_file)
        #get facial feautures, namely hair color
        faces = anime_face_features(image_file)

    if not faces:
        #no faces identified, time to guess
        hair_color = cvim[cvim.shape[0]//4,cvim.shape[1]//2]
    else:
        #grab hair color
        hair_color = color_object_to_list(faces[0].hair.color)

    if clusterLevel > 0:
        # cluster the colors
        preprocessed = clustering(cvim,clusterLevel)
    else:
        preprocessed = cvim

    #create a mask from the hair colors
    mask = color_mask(hair_color, preprocessed,options)

    #reduce the noise in the image
    mask = noise_reduction(mask)

    #trace around the largest areas
    mask = contouring(cvim,mask)

    # add blur to feather/smooth edges
    mask = cv2.GaussianBlur(mask,(21,21),5)

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
    animated_gif = image_to_rainbow_gif(cvim_rgb,mask,output,options=options)

    return(animated_gif)


if __name__ == '__main__':
    filename = '/project/data/downloads/input.png'
    url = 'https://cdn.discordapp.com/attachments/748030642058559510/749078292820262954/kindpng_2748314.png'
    # main(single_file=filename,output_folder=".")
    main(url=url)
