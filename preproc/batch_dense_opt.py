import os
import sys
import cv2
import argparse
import numpy as np
import warnings

def main():
    warnings.simplefilter("error")

    parser = argparse.ArgumentParser(description="optical flow")
    parser.add_argument("--start", type=int, help="start id")
    parser.add_argument("--end", type=int, help="end id")
    parser.add_argument("--rgb", type=str, help="rgb dir")
    parser.add_argument("--opt", type=str, help="opt dir")
    parser.add_argument("--csv", type=str, help="csv file")
    args = parser.parse_args()

    root_dir, opt_dir, catalog = args.rgb, args.opt, args.csv
    with open(catalog, "r") as fo:
        subdirs = list(map(lambda x: x.strip().split(",")[0], fo.readlines()))
    start, end = args.start, args.end if args.end != -1 else len(subdirs)
    for i, subdir in enumerate(subdirs):
        if not (i >= start and i < end):
            continue
        
        subdir_full = os.path.join(root_dir, subdir)
        imnames = list(map(lambda x: os.path.join(subdir_full, x), sorted(os.listdir(subdir_full))))
        
        #fix some dir that not completely calculate opticalflow (you can delete this line)
        if subdir.split('/')[-1] in os.listdir(os.path.join(opt_dir, subdir.split('/')[0])) :
            continue
            
        print(subdir)
            
        
        if imnames[0].endswith('.jpg') == False:
            print(imnames[0])
            #os.rmdir(imnames[0]) #delete .ipynb_checkpoints when upload data from mac
            continue
        
        frame1 = cv2.imread(imnames[0])
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        bgr = np.zeros_like(frame1)
        opt_name = os.path.join(opt_dir, subdir)
        if not os.path.isdir(opt_name):
            os.makedirs(opt_name)
        opt_name = os.path.join(opt_name, imnames[0].split("/")[-1][:-3]+"jpg")
        cv2.imwrite(opt_name, bgr)

        for i in range(1, len(imnames)):
            frame2 = cv2.imread(imnames[i])
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if mag.max() - mag.min() == 0:
                bgr = np.zeros_like(mag).astype(np.uint8)
            else:
                
                '''
                this line occur error in case of magnitude is infinity
                bgr = (255.0*(mag-mag.min())/float(mag.max()-mag.min())).astype(np.uint8)
                '''
                
                bgr = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            imname = imnames[i].split("/")[-1][:-3]+"jpg"
            opt_name = os.path.join(opt_dir, subdir, imname)
            cv2.imwrite(opt_name, bgr)
            prvs = next
    return

if __name__ == '__main__':
    main()
