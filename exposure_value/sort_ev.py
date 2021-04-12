import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
import xlwt
from xlwt import Workbook

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')
sheet1.write(0, 0, 'Index')
sheet1.write(0, 1, 'Under_index')
sheet1.write(0, 2, 'Over_index')
sheet1.write(0, 3, 'Under_EV')
sheet1.write(0, 4, 'Over_EV')
sheet1.write(0, 5, 'total_images')

scene_no = 361
data_dir = './Dataset/Dataset_Part1'

for no in range(1, scene_no):
    file_p        = os.path.join(data_dir,  str(no))
    file_list     = glob.glob(file_p + '/*.{}'.format('JPG'))
    file_list     = sorted(file_list, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    length_images = len(file_list)
    
    print('no:%d, length_images:%d'%(no, length_images))

    data = []
    for i in range(length_images):
        ldr_img           = cv2.imread(file_list[i])
        avg_color_per_row = np.average(ldr_img, axis=2)
        avg_color         = np.average(avg_color_per_row, axis=0)
        avg_color         = np.average(avg_color, axis=0)
        print("index:", i, "average:", avg_color)
        data.append([i, avg_color])

    sorted_by_second = sorted(data, key=lambda tup: tup[1])  # sort ldr images according to the average color
    sheet1.write(no, 0, no)
    sheet1.write(no, 1, sorted_by_second[0][0]+1)
    sheet1.write(no, 2, sorted_by_second[length_images-1][0]+1)
    sheet1.write(no, 3, sorted_by_second[0][1])
    sheet1.write(no, 4, sorted_by_second[length_images-1][1])
    sheet1.write(no, 5, length_images)
    
    #for i in range(length_images):
    #    sheet1.write(no, i+1, sorted_by_second[i][0]+1)
    #    sheet1.write(no, i+1+length_images, sorted_by_second[i][1])
   

wb.save('exposure_value_part1.xls')