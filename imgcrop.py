#main.py
import cv2
import numpy as np
import os
import replace



def accrop(dataroot, rawdir, skip=False):
    path = dataroot + rawdir
    pathsave = dataroot + "/cropped" #保存图像文件夹
    
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)
    pathsave = pathsave + "/test" #保存图像文件夹
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)


    if not skip:

        for file in os.listdir(path):  #遍历访问图像
            if (len(file)>4 and file[-4] != '.') or len(file)<5:
                pathsub = path + file + "/"
                print('find the raw data in', pathsub)
                
                for filed in os.listdir(pathsub):
                    
                    img_path = (pathsub + filed)
                    img = cv2.imread(img_path)  #读取图像
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    h, w = gray.shape
                    if h > 890 and w > 1590:
                        hm = h-int(h/3)
                        for yi in range(0, w):
                            CC = gray[hm, w-yi-1]

                            if CC < 100:

                                y2 = w-yi-1
                                #print('y2', y2)
                                break
                        
                        for xi in range(0, hm):
                            CC = gray[hm-xi, y2]

                            if CC > 100:

                                x1 = hm-xi
                                #print('x1', x1)
                                break

                        for xi in range(0, hm):
                            CC = gray[hm+xi, y2]

                            if CC > 100:

                                x2 = hm+xi
                                #print('x2', x2)
                                break

                        for yi in range(0, w):
                            CC = gray[x1+1, y2-yi-1]
                            #print(CC)

                            if CC > 100:

                                y1 = y2-yi
                                #print('y1', y1)
                                break

                        cropimage = img[x1+3:x2-3, y1+3:y2-3,:]
                        #cv2.imshow("CROP", cropimage)

                        #cv2.waitKey(0)

                        cv2.imwrite(pathsave+filed, cropimage)

        return pathsave[:-5]



def fastcrop(dataroot, rawdir, region, skip=False):
    
    path = dataroot + rawdir
    pathsave = dataroot + "/cropped" #保存图像文件夹
    
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)
    pathsave = pathsave + "/test" #保存图像文件夹
    if not os.path.exists(pathsave):
        os.mkdir(pathsave)

    pathsave1 = pathsave + "/no_pulsar" #保存图像文件夹
    if not os.path.exists(pathsave1):
        os.mkdir(pathsave1)
    pathsave2 = pathsave + "/pulsar" #保存图像文件夹
    if not os.path.exists(pathsave2):
        os.mkdir(pathsave2)

    returnpathsave = pathsave
    pathsave = pathsave1
    print('!!! The cropped images are saved in', pathsave)
        

    if not skip:
        for file1 in os.listdir(path):  #遍历访问文件/文件夹 in workdir
            pathsub1 = path + "/" + file1 + "/"  
            if os.path.isdir(pathsub1):
                #print(file1)
                #pathsub = path + "/" + file1 + "/"
                print('find the raw data in', pathsub1)

                for file in os.listdir(pathsub1):  #遍历访问文件、文件夹 Dec+xx
                    pathsub = pathsub1 + "/" + file + "/"
                    if os.path.isdir(pathsub):
                        #print(file)
                        #pathsub = path + "/" + file + "/"
                        print('find the raw data in', pathsub)

                        total = len(os.listdir(pathsub))
                        pic = 0

                        for filed in os.listdir(pathsub):
                            
                            img_path = (pathsub + filed)
                            if not os.path.exists(pathsave+filed):
                                img = cv2.imread(img_path)  #读取图像
                                if img.ndim == 3:
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                elif img.ndim == 2:
                                    gray = img

                                h, w = gray.shape
                                if h > 890 and w > 1590:
                                    

                                    cropimage = img[region[0]+2:region[1]-2, region[2]+2:region[3]-2,:]
                                    #cv2.imshow("CROP", cropimage)

                                    #cv2.waitKey(0)

                                    cv2.imwrite(pathsave+"/"+filed, cropimage)

                            pic = pic + 1
                            step = int(100 / total * (pic + 1))
                            str1 = '\r[%3d%%] %s' % (step, '>' * step)
                            print(str1, end='', flush=True)
                            if pic == total:
                                print('cropping is finished! \n')

                    

    return returnpathsave

