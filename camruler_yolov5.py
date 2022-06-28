#-------------------------------
# imports
#-------------------------------

# builtins
import os,sys,time,traceback
from math import hypot
import torch
from time import time


# must be installed using pip
# python3 -m pip install opencv-python
import numpy as np
import cv2
import numpy as np
import imutils

# local clayton libs
import yutils.frame_capture as frame_capture
import yutils.frame_draw as frame_draw

from PIL import Image, ImageDraw, ImageFont

class Detection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self,model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:



                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)


                x1c,y1c = conv(x1-(cx),y1-(cy))
                x2c,y2c = conv(x2-(cx),y2-(cy))
                xlen = abs(x1c-x2c)
                ylen = abs(y1c-y2c)
                alen = 0
                if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                    alen = (xlen+ylen)/2              
                carea = xlen*ylen

                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 1)
                
                # draw.add_text(frame,f'{xlen:.2f}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
                # # draw.add_text(image,f'Area: {carea:.2f}',x3,y2+8,center=True,top=True,color='red')
                # # if alen:
                # #     draw.add_text(image,f'Avg: {alen:.2f}',x3,y2+34,center=True,top=True,color='green')
                # if x1 < width-x2:
                #     draw.add_text(frame,f'{ylen:.2f}',x2+4,(y1+y2)/2,middle=True,color='red')
                # else:
                #     draw.add_text(frame,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')

        return frame

detector = Detection(model_name= 'best5.pt')

#-------------------------------
# camera
#-------------------------------

# get camera id from argv[1]
# example "python3 camruler.py 2"
ip = 'http://192.168.1.2:4747/video'
camera_id = 0
if len(sys.argv) > 1:
    camera_id = sys.argv[1]
    if camera_id.isdigit():
        camera_id = int(camera_id)

# camera thread setup
camera = frame_capture.Camera_Thread()
camera.camera_source = camera_id # SET THE CORRECT CAMERA NUMBER
#camera.camera_width,camera.camera_height =  640, 480
#camera.camera_width,camera.camera_height = 1280, 720
#camera.camera_width,camera.camera_height = 1280,1024
camera.camera_width,camera.camera_height = 1920,1080
camera.camera_frame_rate = 30
#camera.camera_fourcc = cv2.VideoWriter_fourcc(*"YUYV")
camera.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# start camera thread
camera.start()

# initial camera values (shortcuts for below)
width  = camera.camera_width
height = camera.camera_height
area = width*height
cx = int(width/2)
cy = int(height/2)
dm = hypot(cx,cy) # max pixel distance
frate  = camera.camera_frame_rate
print('CAMERA:',[camera.camera_source,width,height,area,frate])

#-------------------------------
# frame drawing/text module 
#-------------------------------

draw = frame_draw.DRAW()
draw.width = width
draw.height = height

#-------------------------------
# conversion (pixels to measure)
#-------------------------------

# distance units designator
unit_suffix = 'um'

# calibrate every N pixels
pixel_base = 10

# maximum field of view from center to farthest edge
# should be measured in unit_suffix 
cal_range = 72

# initial calibration values table {pixels:scale}
# this is based on the frame size and the cal_range
cal = dict([(x,cal_range/dm) for x in range(0,int(dm)+1,pixel_base)])

# calibration loop values
# inside of main loop below
cal_base = 5
cal_last = None

# calibration update
def cal_update(x,y,unit_distance):

    # basics
    pixel_distance = hypot(x,y)
    scale = abs(unit_distance/pixel_distance)
    target = baseround(abs(pixel_distance),pixel_base)

    # low-high values in distance
    low  = target*scale - (cal_base/2)
    high = target*scale + (cal_base/2)

    # get low start point in pixels
    start = target
    if unit_distance <= cal_base:
        start = 0
    else:
        while start*scale > low:
            start -= pixel_base

    # get high stop point in pixels
    stop = target
    if unit_distance >= baseround(cal_range,pixel_base):
        high = max(cal.keys())
    else:
        while stop*scale < high:
            stop += pixel_base

    # set scale
    for x in range(start,stop+1,pixel_base):
        cal[x] = scale
        print(f'CAL: {x} {scale}')

# read local calibration data
calfile = 'camruler_cal.csv'
if os.path.isfile(calfile):
    with open(calfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] in ('d',):
                axis,pixels,scale = [_.strip() for _ in line.split(',',2)]
                if axis == 'd':
                    print(f'LOAD: {pixels} {scale}')
                    cal[int(pixels)] = float(scale)

# convert pixels to units
def conv(x,y):

    d = distance(0,0,x,y)

    scale = cal[baseround(d,pixel_base)]

    return x*scale,y*scale

# round to a given base
def baseround(x,base=1):
    return int(base * round(float(x)/base))

# distance formula 2D
def distance(x1,y1,x2,y2):
    return hypot(x1-x2,y1-y2)

#-------------------------------
# define frames
#-------------------------------

# define display frame
framename = "Morphometric Analysis"
cv2.namedWindow(framename,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)

#-------------------------------
# key events
#-------------------------------

key_last = 0
key_flags = {'config':False, # c key
             'auto':False,   # a key
             'thresh':False, # t key
             'percent':False,# p key
             'norms':False,  # n key
             'rotate':False, # r key
             'lock':False,   # 
             'capture': False,
             }

def key_flags_clear():

    global key_flags

    for key in list(key_flags.keys()):
        if key not in ('rotate',):
            key_flags[key] = False

def key_event(key):

    global key_last
    global key_flags
    global mouse_mark
    global cal_last

    # config mode
    if key == 99:
        if key_flags['config']:
            key_flags['config'] = False
        else:
            key_flags_clear()
            key_flags['config'] = True
            cal_last,mouse_mark = 0,None

    # normilization mode
    elif key == 110:
        if key_flags['norms']:
            key_flags['norms'] = False
        else:
            key_flags['thresh'] = False
            key_flags['percent'] = False
            key_flags['lock'] = False
            key_flags['norms'] = True
            mouse_mark = None

    # rotate
    elif key == 114:
        if key_flags['rotate']:
            key_flags['rotate'] = False
        else:
            key_flags['rotate'] = True

    # auto mode
    elif key == 97:
        if key_flags['auto']:
            key_flags['auto'] = False
        else:
            key_flags_clear()
            key_flags['auto'] = True
            mouse_mark = None

    # auto percent
    elif key == 112 and key_flags['auto']:
        key_flags['percent'] = not key_flags['percent']
        key_flags['thresh'] = False
        key_flags['lock'] = False

    # auto threshold
    elif key == 116 and key_flags['auto']:
        key_flags['thresh'] = not key_flags['thresh']
        key_flags['percent'] = False
        key_flags['lock'] = False

    elif key == 32:
        # if key_flags['auto']:
        cv2.imwrite('object-measurement/imags/imags.png', frame0)


    # log
    print('key:',[key,chr(key)])
    key_last = key
    
#-------------------------------
# mouse events
#-------------------------------

# mouse events
mouse_raw  = (0,0) # pixels from top left
mouse_now  = (0,0) # pixels from center
mouse_mark = None  # last click (from center)

# auto measure mouse events
auto_percent = 0.2 
auto_threshold = 127
auto_blur = 5

# normalization mouse events
norm_alpha = 0
norm_beta = 255

# mouse callback
def mouse_event(event,x,y,flags,parameters):

    #print(event,x,y,flags,parameters)

    # event =  0 = current location
    # event =  1 = left   down click
    # event =  2 = right  down click
    # event =  3 = middle down
    # event =  4 = left   up   click
    # event =  5 = right  up   click
    # event =  6 = middle up
    # event = 10 = middle scroll, flag negative|positive value = down|up

    # globals
    global mouse_raw
    global mouse_now
    global mouse_mark
    global key_last
    global auto_percent
    global auto_threshold
    global auto_blur
    global norm_alpha
    global norm_beta

    # update percent
    if key_flags['percent']:
        auto_percent = 5*(x/width)*(y/height)

    # update threshold
    elif key_flags['thresh']:
        auto_threshold = int(255*x/width)
        auto_blur = int(20*y/height) | 1 # insure it is odd and at least 1

    # update normalization
    elif key_flags['norms']:
        norm_alpha = int(64*x/width)
        norm_beta  = min(255,int(128+(128*y/height)))

    # update mouse location
    mouse_raw = (x,y)

    # offset from center
    # invert y to standard quadrants
    ox = x - cx
    oy = (y-cy)*-1

    # update mouse location
    mouse_raw = (x,y)
    if not key_flags['lock']:
        mouse_now = (ox,oy)

    # left click event
    if event == 1:

        if key_flags['config']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)

        elif key_flags['auto']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)

        if key_flags['percent']:
            key_flags['percent'] = False
            mouse_mark = (ox,oy)
            
        elif key_flags['thresh']:
            key_flags['thresh'] = False
            mouse_mark = (ox,oy)
            
        elif key_flags['norms']:
            key_flags['norms'] = False
            mouse_mark = (ox,oy)

        elif not key_flags['lock']:
            if mouse_mark:
                key_flags['lock'] = True
            else:
                mouse_mark = (ox,oy)
        else:
            key_flags['lock'] = False
            mouse_now = (ox,oy)
            mouse_mark = (ox,oy)

        key_last = 0

    # right click event
    elif event == 2:
        key_flags_clear()
        mouse_mark = None
        key_last = 0

# register mouse callback
cv2.setMouseCallback(framename,mouse_event)

#-------------------------------
# main loop
#-------------------------------

# loop
while 1:

    # get frame
    frame0 = camera.next(wait=1)
    if frame0 is None:
        time.sleep(0.1)
        continue

    # normalize
    cv2.normalize(frame0,frame0,norm_alpha,norm_beta,cv2.NORM_MINMAX)

    # rotate 180
    if key_flags['rotate']:
        frame0 = cv2.rotate(frame0,cv2.ROTATE_180)

    # start top-left text block
    text = []

    # camera text
    fps = camera.current_frame_rate
    text.append(f'CAMERA: {camera_id} {width}x{height} {fps}FPS')

    # mouse text
    text.append('')
    if not mouse_mark:
        text.append(f'LAST CLICK: NONE')
    else:
        text.append(f'LAST CLICK: {mouse_mark} PIXELS')
    text.append(f'CURRENT XY: {mouse_now} PIXELS')

    #-------------------------------
    # normalize mode
    #-------------------------------
    if key_flags['norms']:

        # print
        text.append('')
        text.append(f'NORMILIZE MODE')
        text.append(f'ALPHA (min): {norm_alpha}')
        text.append(f'BETA (max): {norm_beta}')
        
    #-------------------------------
    # config mode
    #-------------------------------
    if key_flags['config']:

        # quadrant crosshairs
        draw.crosshairs(frame0,5,weight=2,color='red',invert=True)

        # crosshairs aligned (rotated) to maximum distance 
        draw.line(frame0,cx,cy, cx+cx, cy+cy,weight=1,color='red')
        draw.line(frame0,cx,cy, cx+cy, cy-cx,weight=1,color='red')
        draw.line(frame0,cx,cy,-cx+cx,-cy+cy,weight=1,color='red')
        draw.line(frame0,cx,cy, cx-cy, cy+cx,weight=1,color='red')

        # mouse cursor lines (parallel to aligned crosshairs)
        mx,my = mouse_raw
        draw.line(frame0,mx,my,mx+dm,my+(dm*( cy/cx)),weight=1,color='green')
        draw.line(frame0,mx,my,mx-dm,my-(dm*( cy/cx)),weight=1,color='green')
        draw.line(frame0,mx,my,mx+dm,my+(dm*(-cx/cy)),weight=1,color='green')
        draw.line(frame0,mx,my,mx-dm,my-(dm*(-cx/cy)),weight=1,color='green')
    
        # config text data
        text.append('')
        text.append(f'CONFIG MODE')

        # start cal
        if not cal_last:
            cal_last = cal_base
            caltext = f'CONFIG: Click on D = {cal_last}'

        # continue cal
        elif cal_last <= cal_range:
            if mouse_mark:
                cal_update(*mouse_mark,cal_last)
                cal_last += cal_base
            caltext = f'CONFIG: Click on D = {cal_last}'

        # done
        else:
            key_flags_clear()
            cal_last == None
            with open(calfile,'w') as f:
                data = list(cal.items())
                data.sort()
                for key,value in data:
                    f.write(f'd,{key},{value}\n')
                f.close()
            caltext = f'CONFIG: Complete.'

        # add caltext
        draw.add_text(frame0,caltext,(cx)+100,(cy)+30,color='red')

        # clear mouse
        mouse_mark = None     

    #-------------------------------
    # auto mode
    #-------------------------------
    elif key_flags['auto']:
        
        mouse_mark = None

        # auto text data
        text.append('')
        text.append(f'AUTO MODE')
        text.append(f'UNITS: {unit_suffix}')

        
        # gray frame
        frame1 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)

        # blur frame
        frame1 = cv2.GaussianBlur(frame1,(auto_blur,auto_blur),0)

        # threshold frame n out of 255 (85 = 33%)
        frame1 = cv2.threshold(frame1,auto_threshold,255,cv2.THRESH_BINARY)[1]

        # invert
        frame1 = ~frame1

        
        # detector()
        # load_model('non_essentials\yolo\best5.pt')

        start_time = time()
        results = detector.score_frame(frame0)
        frame = detector.plot_boxes(results, frame0)
        
        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)
        #print(f"Frames Per Second : {fps}")
            
        cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 1)

        if cv2.waitKey(1) & 0xFF == ord('z'):
            cv2.imwrite(f'assets/captured_img/img_cap.png', frame0)

            im = Image.open('assets/captured_img/img_cap.png')
            width, height = im.size

            drawer = ImageDraw.Draw(im)
            texty = "photograpped by Andrei Cyril Gimoros - this image is a property of Philippine Nuclear Research Institute"

            font = ImageFont.truetype('arial.ttf', 20)
            textwidth, textheight = drawer.textsize(texty, font)

            # calculate the x,y coordinates of the text
            margin = 10
            x = width - textwidth - margin
            y = height - textheight - margin

            # draw watermark in the bottom right corner
            drawer.text((x, y), texty, font=font)
            im.show()
            im.save('assets/captured_img/pred_watermark.jpg')

        if cv2.waitKey(1) & 0xFF == ord('p'):
            arr = os.listdir("dataset/val_copy/")
            for i in arr:                    

                predicted_img = cv2.imread(f'dataset/val_copy/{i}')
                results = detector.score_frame(predicted_img)
                framex = detector.plot_boxes(results, predicted_img)
                # cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 1)

                cv2.imwrite(f'assets/pred2/{i}', predicted_img)
            
        # template1 = cv2.imread("object-measurement/retinaculum.png")
        # template1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
        # template1 = cv2.Canny(template1, 50, 200)
        # template = imutils.resize(template1, width=60)
        # (tH, tW) = template.shape[:2]

        # gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        # found = None

        # # loop over the scales of the image
        # for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        #     # resize the image according to the scale, and keep track
        #     # of the ratio of the resizing
        #     resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        #     r = gray.shape[1] / float(resized.shape[1])

        #     # if the resized image is smaller than the template, then break
        #     # from the loop
        #     if resized.shape[0] < tH or resized.shape[1] < tW:
        #         print("frame is smaller than the template")
        #         break

        #     # detect edges in the resized, grayscale image and apply template
        #     # matching to find the template in the image
        #     edged = cv2.Canny(resized, 50, 200)
        #     result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        #     (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        #     # if we have found a new maximum correlation value, then update
        #     # the bookkeeping variable
        #     if found is None or maxVal > found[0]:
        #         found = (maxVal, maxLoc, r)

        #     # unpack the bookkeeping variable and compute the (x, y) coordinates
        #     # of the bounding box based on the resized ratio
        # # print(found)
        # if found is None:
        #     # just show only the frames if the template is not detected
        #     # cv2.imshow(frame0)
        #     print("No template is found")
        # else:
        #     (_, maxLoc, r) = found
        #     (x1, y1) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        #     (x2, y2) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        #     (x3,y3) = x1+(tW/2),y1+(tH/2)
        #     print(x1,y1,x2,y2)

        #     # draw a bounding box around the detected result and display the image
        #     # cv2.rectangle(frame0, (startX, startY), (endX, endY), (0, 0, 255), 2)
        #     x1c,y1c = conv(x1-(cx),y1-(cy))
        #     x2c,y2c = conv(x2-(cx),y2-(cy))
        #     xlen = abs(x1c-x2c)
        #     ylen = abs(y1c-y2c)
        #     alen = 0
        #     if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
        #         alen = (xlen+ylen)/2              
        #     carea = xlen*ylen

        #     draw.rect(frame0,x1,y1,x2,y2,weight=2,color='red')

        #     draw.add_text(frame0,f'{xlen:.2f}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
        #     draw.add_text(frame0,f'Area: {carea:.2f}',x3,y2+8,center=True,top=True,color='red')
        #     if alen:
        #         draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y2+34,center=True,top=True,color='green')
        #     if x1 < width-x2:
        #         draw.add_text(frame0,f'{ylen:.2f}',x2+4,(y1+y2)/2,middle=True,color='red')
        #     else:
        #         draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')

            # cv2.imshow(frame0)





        # find contours on thresholded image
        # contours,nada = cv2.findContours(frame1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # # small crosshairs (after getting frame1)
        # draw.crosshairs(frame0,5,weight=2,color='green')    
    
        # # loop over the contours
        # for c in contours:

        #     # contour data (from top left)
        #     x1,y1,w,h = cv2.boundingRect(c)
        #     x2,y2 = x1+w,y1+h
        #     x3,y3 = x1+(w/2),y1+(h/2)

        #     # percent area
        #     percent = 100*w*h/area
            
        #     # if the contour is too small, ignore it
        #     if percent < auto_percent:
        #             continue

        #     # if the contour is too large, ignore it
        #     elif percent > 60:
        #             continue

        #     # convert to center, then distance
        #     x1c,y1c = conv(x1-(cx),y1-(cy))
        #     x2c,y2c = conv(x2-(cx),y2-(cy))
        #     xlen = abs(x1c-x2c)
        #     ylen = abs(y1c-y2c)
        #     alen = 0
        #     if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
        #         alen = (xlen+ylen)/2              
        #     carea = xlen*ylen

        #     # plot
        #     draw.rect(frame0,x1,y1,x2,y2,weight=2,color='red')

        #     # add dimensions
        #     draw.add_text(frame0,f'{xlen:.2f}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
        #     draw.add_text(frame0,f'Area: {carea:.2f}',x3,y2+8,center=True,top=True,color='red')
        #     if alen:
        #         draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y2+34,center=True,top=True,color='green')
        #     if x1 < width-x2:
        #         draw.add_text(frame0,f'{ylen:.2f}',x2+4,(y1+y2)/2,middle=True,color='red')
        #     else:
        #         draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')

    #-------------------------------
    # dimension mode
    #-------------------------------
    else:

        # small crosshairs
        draw.crosshairs(frame0,5,weight=2,color='green')    

        # mouse cursor lines
        draw.vline(frame0,mouse_raw[0],weight=1,color='green')
        draw.hline(frame0,mouse_raw[1],weight=1,color='green')
       
        # draw
        if mouse_mark:

            # locations
            x1,y1 = mouse_mark
            x2,y2 = mouse_now

            # convert to distance
            x1c,y1c = conv(x1,y1)
            x2c,y2c = conv(x2,y2)
            xlen = abs(x1c-x2c)
            ylen = abs(y1c-y2c)
            llen = hypot(xlen,ylen)
            alen = 0
            if max(xlen,ylen) > 0 and min(xlen,ylen)/max(xlen,ylen) >= 0.95:
                alen = (xlen+ylen)/2              
            carea = xlen*ylen

            # print distances
            text.append('')
            text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
            text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
            text.append(f'L LEN: {llen:.2f}{unit_suffix}')

            # convert to plot locations
            x1 += cx
            x2 += cx
            y1 *= -1
            y2 *= -1
            y1 += cy
            y2 += cy
            x3 = x1+((x2-x1)/2)
            y3 = max(y1,y2)

            # line weight
            weight = 1
            if key_flags['lock']:
                weight = 2

            # plot
            draw.rect(frame0,x1,y1,x2,y2,weight=weight,color='red')
            draw.line(frame0,x1,y1,x2,y2,weight=weight,color='green')

            # add dimensions
            draw.add_text(frame0,f'{xlen:.2f}',x1-((x1-x2)/2),min(y1,y2)-8,center=True,color='red')
            draw.add_text(frame0,f'Area: {carea:.2f}',x3,y3+8,center=True,top=True,color='red')
            if alen:
                draw.add_text(frame0,f'Avg: {alen:.2f}',x3,y3+34,center=True,top=True,color='green')           
            if x2 <= x1:
                draw.add_text(frame0,f'{ylen:.2f}',x1+4,(y1+y2)/2,middle=True,color='red')
                draw.add_text(frame0,f'{llen:.2f}',x2-4,y2-4,right=True,color='green')
            else:
                draw.add_text(frame0,f'{ylen:.2f}',x1-4,(y1+y2)/2,middle=True,right=True,color='red')
                draw.add_text(frame0,f'{llen:.2f}',x2+8,y2-4,color='green')

    # add usage key
    text.append('')
    text.append(f'Q = QUIT')
    # text.append(f'R = ROTATE')
    # text.append(f'N = NORMALIZE')
    text.append(f'A = AUTO-MODE')
    # if key_flags['auto']:
    #     text.append(f'P = MIN-PERCENT')
    #     text.append(f'T = THRESHOLD')
    #     text.append(f'T = GAUSS BLUR')
    text.append(f'C = CONFIG-MODE')
    


    # draw top-left text block
    draw.add_text_top_left(frame0,text)

    # display
    cv2.imshow(framename,frame0)

    # key delay and action
    key = cv2.waitKey(1) & 0xFF

    # esc ==  27 == quit
    # q   == 113 == quit
    if key in (27,113):
        break

    # key data
    #elif key != 255:
    elif key not in (-1,255):
        key_event(key)

    # elif key in (32):
    #     cv2.imwrite(f'object-deetector/imags/cap.png', frame0)
#-------------------------------
# kill sequence
#-------------------------------

# close camera thread
camera.stop()

# close all windows
cv2.destroyAllWindows()

# done
exit()

#-------------------------------
# end
#-------------------------------
