from __future__ import print_function
import glob, os, cv2, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd

p      = './exposure_value/exposure_value_part1.xls'
wb     = xlrd.open_workbook(p)
sheet1 = wb.sheet_by_index(0)

patch_size   = 256 
batch_size   = 20
patch_stride = 256 

out_dir      = './Dataset/tfrecord/'
data_dir     = './Dataset/train/'

scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if scene_dir!="Label"]
scene_dirs = sorted(scene_dirs, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
num_scenes = len(scene_dirs)

count            = 0
cur_writing_path = os.path.join(out_dir, "train_{:d}_{:04d}.tfrecords".format(patch_stride, 0))
writer           = tf.python_io.TFRecordWriter(cur_writing_path)

def norm_0_to_1(img):
    img       = np.float32(img)
    img_flat  = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img   = (img - min_value) * 1 / (max_value - min_value)
    return new_img

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_random(over_exp, under_exp, label, patch_size):
    h, w, c = np.shape(over_exp)

    def write_example(h1, h2, w1, w2):
        global count
        global writer

        cur_batch_index = count // batch_size

        if count % batch_size == 0:
            writer.close()
            cur_writing_path = os.path.join(out_dir,
                                            "train_{:d}_{:04d}.tfrecords".format(patch_stride, cur_batch_index))
            writer = tf.python_io.TFRecordWriter(cur_writing_path)

        over_exp_patch  = over_exp[h1:h2, w1:w2, ::-1]
        under_exp_patch = under_exp[h1:h2, w1:w2, ::-1]
        in_LDR_patch    = np.concatenate([under_exp_patch, over_exp_patch], axis=2)
        ref_HDR_patch   = label[h1:h2, w1:w2, ::-1]

        count += 1

        if count % 1000 == 0:
             plt.figure(1)
             plt.subplot(311).set_title('over_exp')
             plt.imshow(over_exp_patch)
             plt.subplot(312).set_title('under_exp')
             plt.imshow(under_exp_patch)
             plt.subplot(313).set_title('ref_HDR')
             plt.imshow(ref_HDR_patch)
             plt.show()
            
        # create example
        example = tf.train.Example(features=tf.train.Features(feature={
            'in_LDR': bytes_feature(in_LDR_patch.tostring()),
            'ref_HDR': bytes_feature(ref_HDR_patch.tostring()),
        }))
        writer.write(example.SerializeToString())

    # generate patches
    for h_ in range(0, h - patch_size + 1, patch_stride):
        for w_ in range(0, w - patch_size + 1, patch_stride):
            write_example(h_, h_ + patch_size, w_, w_ + patch_size)

    # deal with border patch
    if h % patch_size:
        for w_ in range(0, w - patch_size + 1, patch_stride):
            write_example(h - patch_size, h, w_, w_ + patch_size)

    if w % patch_size:
        for h_ in range(0, h - patch_size + 1, patch_stride):
            write_example(h_, h_ + patch_size, w - patch_size, w)

    if w % patch_size and h % patch_size:
        write_example(h - patch_size, h, w - patch_size, w)


if __name__ == "__main__":
    """####################### image reading ############################"""
    for index in range(num_scenes):
        start_img_time = time.time()
        cur_path       = scene_dirs[index]
        if cur_path=="Label":
          continue
        cur_path       = os.path.join(data_dir, cur_path)

        # Read HDR LDR images
        file_list     = glob.glob(cur_path + '/*.{}'.format('JPG'))
        file_list     = sorted(file_list, key=lambda i: int(os.path.splitext(os.path.basename(i))[0])) 
        length_images = len(file_list)

        no          = int(scene_dirs[index])
        under_index = sheet1.cell_value(no, 1)
        over_index  = sheet1.cell_value(no, 2)
        under_index = int(under_index)
        over_index  = int(over_index)
        print('no:%d, over:%d, under:%d'%(no, over_index, under_index))

        over_exp  = cv2.imread(file_list[over_index-1])
        under_exp = cv2.imread(file_list[under_index-1])

        # finding corresponding ldr image
        label_p = os.path.join(data_dir, 'Label', scene_dirs[index]+'.JPG')
        label   = cv2.imread(label_p)
        ''' check shape '''
        assert (np.shape(over_exp) == np.shape(label))
        assert (np.shape(over_exp) == np.shape(under_exp))

        ''' bring to [0,1] '''
        over_exp  = norm_0_to_1(over_exp)
        under_exp = norm_0_to_1(under_exp)
        label     = norm_0_to_1(label)
        
        '''############################ cropping images ############################'''
        crop_random(over_exp, under_exp, label, patch_size)
        elapsed_img = time.time() - start_img_time

        print('Processed Image ->' + cur_path, ' %d / %d, took: %s' % (index + 1, num_scenes, elapsed_img)) 

    writer.close()
    print("Finished!\nTotal number of patches:", count)
