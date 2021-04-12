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
data_dir = r'G:\PROJECTS\FPGA\Fusion Dataset\Cai2018\Dataset_Part1\Dataset_Part1'

for no in range(1, scene_no):
    file_p = os.path.join(data_dir,  str(no))
    file_list = glob.glob(file_p + '/*.{}'.format('JPG'))
    file_list = sorted(file_list, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
    length_images = len(file_list)
    #results_p = os.path.join('G:\PROJECTS\FPGA\Fusion Dataset\Cai2018\Dataset_Part2\Dataset_Part2_cp', str(no))

    print('no:%d, length_images:%d'%(no, length_images))

    data = []
    for i in range(length_images):
        ldr_img = cv2.imread(file_list[i])
        print(file_list[i])
        avg_color_per_row = np.average(ldr_img, axis=2)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color = np.average(avg_color, axis=0)
        print("average:", avg_color, "index:", i)
        data.append([i, avg_color])

    sorted_by_second = sorted(data, key=lambda tup: tup[1])  # sort ldr images according to the average color

    sheet1.write(no, 0, no)
    
    for i in range(length_images):
        sheet1.write(no, i+1, sorted_by_second[i][0]+1)
        sheet1.write(no, i+1+length_images, sorted_by_second[i][1])
   

wb.save('exposure_value_part1_all.xls')
    # rewrite images according to its EV
    # for i in range(length_images):
    #     plt.subplot(5, max(length_images // 4 + 1, 1), i + 1).set_title(str(i + 1))
    #     ind = sorted_by_second[i][0]
    #     p = file_list[ind]
    #     im = cv2.imread(p)
    #     plt.imshow(im)
    # plt.show()

    # for i in range(length_images):
    #     ev = cv2.imread(file_list[sorted_by_second[i][0]])
    #     im_p = os.path.join(results_p, str(i + 1) + '.png')
    #     if not os.path.exists(results_p):
    #         os.makedirs(results_p)
    #     cv2.imwrite(im_p, ev)
    #
    #     # Convert 2 images to numpy arrays and compare pixel by pixel
    #     ev_cp = cv2.imread(im_p)
    #     flag = (ev == ev_cp).all()
    #     #print(np.where(flag==False))
    #     print(flag)





