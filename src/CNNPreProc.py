
import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from cv2 import imread
from glob import glob
import scipy.misc
import os

# The path where all the files will be stored
path="C:\\Users\\nihal\\PycharmProjects\\datascp\\data\\"
# load the images
dno=0
with open("Label.txt", "a") as myfile:
    for d in os.listdir(path):
        print("Doing "+str(d))
        imgs=imread(path+"\\"+d+"\\"+'images\\image00000.tiff')
        files = sorted(glob(path+"\\"+d+"\\"+'images\\*.tiff'))
        file_count=0
        for f in files:
            print("Processing Image and its Regions :"+f,end="\r")
            imgs+=imread(f) # add each value of a pixel of all images here 
            file_count+=1
        rgb_min = 0
        rgb_max = 1
        normal_min = 0
        normal_max = 255
        
        # Normaliztion from 0 to 1
        imgs= rgb_min+(((imgs-normal_min)*(rgb_max-rgb_min))/(normal_max-normal_min)) 
        fname='outfile_'+str(dno)+'.jpg' 
        
        # Save the flatten from 4D to 3D to a image
        scipy.misc.imsave('outfile_'+str(dno)+'.jpg', imgs)
        dno += 1


        with open(path + "\\" + d + "\\" + "regions\\regions.json") as f:
            regions = json.load(f)
            i = 0
            squareregions = {}
            # The bounding box creation for each neuron 
            for rr in regions:
                maxx = 0
                maxy = 0
                minx = 512
                miny = 512
                # print("id: "+str(regions[i]["id"]))
                for r in regions[i]["coordinates"]:
                    if maxx < r[0]:
                        maxx = r[0]
                    if maxy < r[1]:
                        maxy = r[1]
                    if minx > r[0]:
                        minx = r[0]
                    if miny > r[1]:
                        miny = r[1]
                arry = []
                top = [minx, miny]
                bottom = [maxx, maxy]
                arry.append(top)
                arry.append(bottom)
                # print(arry)
                # squareregions[regions[i]["id"]]=arry
                i += 1
                # Append to label file
                myfile.write(path +fname+ "," + str(minx) + "," + str(miny) + "," + str(maxx) + "," + str(maxy) + "," + "neuron\n")



# show the outputs
plt.imshow(imgs)
plt.show()
