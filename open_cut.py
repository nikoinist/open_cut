import cv2
import numpy as np
from PIL import Image
import os
from shutils import resize
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required = True, help="Working directory")
ap.add_argument("-r", "--rotate", action='store_true', default=False, help="preform rotation when cutting")
#ap.add_argument("-c", "--cut_size", default=100, type=int , help="size of image to save when cut, in px")
args = vars(ap.parse_args())

working_dir = args["dir"]
path = os.listdir(path = working_dir)

# funkcija za obradu i izrezivanje sekcija karte
# func for processing and cutting map image sections
def open_cut(path, name):
    
    #loading the image
    
    img= cv2.imread(path+name)
    #resizing the image to more managable and less memory heavy pixel size
    resized = resize(img, width=900)
    #plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    #plt.show()
    
    #information about the image size-shape and scale factor used to reduce its size in pixels
    ry = img.shape[1] / float(resized.shape[1])
    rx  = img.shape[0] / float(resized.shape[0])
    
    # Converting and splitting the image into different color channels
    # used for experimenting, final solution uses grayscale and "s" channel from hls color code
    hls = cv2.cvtColor(resized, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #eq = cv2.equalizeHist(gray)

    h= hls[:,:,0]
    l= hls[:,:,1]
    s= hls[:,:,2]

    #l1 = lab[:,:,0]
    #a = lab[:,:,1]
    #b= lab[:,:,2]
    #eqh = cv2.equalizeHist(h)
    #eql = cv2.equalizeHist(l)
    #eqs = cv2.equalizeHist(s)
    
    # Processing algorithm for:
    # - threshold and extract segments from the image
    # - using Sobel algorithm to find y-axis lines in an image "s" and grayscale channels
    # - combine the processed channels, add smoothing, perform thresholding,
    # - close gaps between neighbouring lines and build blocks of thresholded pixels
    # - calculate the contours, calculate the bounding box for the largest contours
    # - exctract the "roi"(Range of interest) from the original image,
    # - transform the extracted area from RGB space to indexed color space(reduced image size)
    # - save the each exctracted area as a new image

    schX = cv2.Sobel(gray, cv2.CV_32F,1,0, scale=1, ksize=-1)
    schY = cv2.Sobel(s, cv2.CV_32F,1,0, scale=1, ksize=-1)
    gradub = cv2.subtract(schX, schY)
    gradub = cv2.convertScaleAbs(gradub)
    sch_x = cv2.convertScaleAbs(schX)
    sch_y = cv2.convertScaleAbs(schY)
    dst = cv2.addWeighted(sch_x,0.3,sch_y,0.1,0)
    sobelCombined = cv2.bitwise_or(sch_x, sch_y)
    blurdst = cv2.GaussianBlur(dst, (7, 7), 1)
    (T, treshG) = cv2.threshold(blurdst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #(T, treshC) = cv2.threshold(sch_y, 0, 900, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #print(T)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    closed = cv2.morphologyEx(treshG, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations =4)
    closed = cv2.dilate(closed, None, iterations = 4)
    cls = closed.copy()
    (_,cnts,_) = cv2.findContours(cls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = False)
    # compute the rotated bounding box of the largest contour
    i=0

    

    def make_directory(path):
        if not os.path.isdir(path):
            print ("pravim direktorij {0}".format(path))
            os.makedirs(path)
        else:
            return

    for cn in c:
        rect = cv2.minAreaRect(cn)
        box = np.int0(cv2.boxPoints(rect))
        (x, y, w, h) = cv2.boundingRect(cn)
        #print(x*rx,y*ry,w*rx,h*ry)


        #v2
        #roi = img[int(y*ry):int(y*ry) + int(h*ry)+100, int(x*rx):int(x*rx) + int(w*rx)+100].copy()
        roi = img[int(y*ry):int(y*ry) + int(h*ry), int(x*rx):int(x*rx) + int(w*rx)].copy()

        # draw a bounding box arounded the detected area and display the
        # image
        #cv2.drawContours(resized, [box], -1, (0, 255, 0), 3)
        #plt.imshow(cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR))

        #plotani izrezani djelovi

        #plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        #plt.show()
        #print(i)
        
        i+=1
        #scale_roi = resize(roi,width=800)
        #final = Image.fromarray(cv2.cvtColor(scale_roi, cv2.COLOR_BGR2RGB))
        
        final = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        pl_index = final.quantize(colors=256, method=0, kmeans=0)
        
		
        #########  OPREZ KOD IZRADE WMS SASTAVNICA DA SE ISKLJUCI ROTACIJA
        # saving processed image
        #print(pl_index.width)
        if args["rotate"] == True:
            make_directory(working_dir+"processed")
            if pl_index.width  > 500:
                rot = pl_index.transpose(Image.ROTATE_270)
                rot.save(working_dir+"processed/"+name[0:-4]+"-{0}.tif".format(i), format='tiff', dpi=(200,200))
        else:
            make_directory(working_dir+"wms")
            if pl_index.width  > 100:
                pl_index.save(working_dir+"wms/"+name[0:-4]+"-{0}.tif".format(i), format='tiff', dpi=(200,200))
        #pl_index.save(working_dir+"wms/"+name[0:-4]+"-{0}.pdf".format(i), format='PDF', dpi=100, resolution=100)
        #cv2.imwrite("02/a-a-{0}.tif".format(i), final)
        #cv2.imwrite("02/a-a-{0}.tif".format(i), final)
        


   
for name in path:
    if name.lower().endswith(('.tif')) or name.lower().endswith(('.tiff')):
        print(name)
        open_cut(working_dir, name)
	
