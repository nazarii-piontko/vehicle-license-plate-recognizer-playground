import os
import shutil
import sys
from pathlib import Path
from random import shuffle
from shutil import copyfile

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def prepare_data_for_yolo_training():
    im_files = []

    if os.path.exists('data/yolo/data'):
        shutil.rmtree('data/yolo/data')
    os.mkdir('data/yolo/data')

    for current_file in Path('data/origin/openalpr').rglob('*.jpg'):
        orig_im_file_path = str(current_file)
        orig_meta_file_path = orig_im_file_path[:-3] + 'txt'

        _, im_file_name = os.path.split(orig_im_file_path)

        im_file_path = 'data/yolo/data/' + im_file_name
        meta_meta_file_path = 'data/yolo/data/' + im_file_name[:-3] + 'txt'

        copyfile(orig_im_file_path, im_file_path)

        im_w, im_h = get_image_size(orig_im_file_path)
        plate_x, plate_y, plate_w, plate_h = get_plate_loc_from_origin(orig_meta_file_path)

        with open(meta_meta_file_path, 'w', newline='\n') as f:
            x = (float(plate_x) + float(plate_w) / 2)/float(im_w)
            y = (float(plate_y) + float(plate_h) / 2) / float(im_h)
            w = float(plate_w)/float(im_w)
            h = float(plate_h)/float(im_h)
            f.write(f'0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

        im_files.append(im_file_path)
     
    for current_file in Path('data/origin/noplates').rglob('*.jpg'):
        orig_im_file_path = str(current_file)

        _, im_file_name = os.path.split(orig_im_file_path)

        im_file_path = 'data/yolo/data/' + im_file_name
        meta_file_path = 'data/yolo/data/' + im_file_name[:-3] + 'txt'

        copyfile(orig_im_file_path, im_file_path)

        with open(meta_file_path, 'w', newline='\n') as f:
            pass

        im_files.append(im_file_path)

    for current_file in Path('data/origin/my').rglob('*.jpg'):
        orig_im_file_path = str(current_file)
        orig_meta_file_path = orig_im_file_path[:-3] + 'txt'

        _, im_file_name = os.path.split(orig_im_file_path)

        im_file_path = 'data/yolo/data/' + im_file_name
        meta_file_path = 'data/yolo/data/' + im_file_name[:-3] + 'txt'

        copyfile(orig_im_file_path, im_file_path)
        copyfile(orig_meta_file_path, meta_file_path)

        im_files.append(im_file_path)

    shuffle(im_files)

    train_len = int((len(im_files) * 6) / 7)
    im_train_files = im_files[0:train_len]
    im_test_files = im_files[train_len:]

    with open('data/yolo/train.txt', 'w', newline='\n') as f:
        for l in im_train_files:
            f.write(l)
            f.write('\n')

    with open('data/yolo/test.txt', 'w', newline='\n') as f:
        for l in im_test_files:
            f.write(l)
            f.write('\n')


def get_plate_loc_from_origin(orig_meta_file_path):
    with open(orig_meta_file_path, 'r') as f:
        orig_meta = f.read()
        orig_meta_parts = orig_meta.split('\t')
        plate_x = int(orig_meta_parts[1])
        plate_y = int(orig_meta_parts[2])
        plate_w = int(orig_meta_parts[3])
        plate_h = int(orig_meta_parts[4])
        return plate_x, plate_y, plate_w, plate_h


def get_image_size(im_file_path):
    im = Image.open(im_file_path)
    return im.size[0], im.size[1]


def show_original_image_with_plate(file_path):
    meta_file_path = file_path[:-3] + 'txt'
    x, y, w, h = get_plate_loc_from_origin(meta_file_path)

    fig, ax = plt.subplots()

    img = plt.imread(file_path)
    ax.imshow(img)

    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))

    plt.show()


def show_train_image_with_plate(im_path):
    meta_file_path = im_path[:-3] + 'txt'

    with open(meta_file_path, 'r') as f:
        orig_meta = f.read()
        orig_meta_parts = orig_meta.split(' ')
        x = float(orig_meta_parts[1])
        y = float(orig_meta_parts[2])
        w = float(orig_meta_parts[3])
        h = float(orig_meta_parts[4])

    fig, ax = plt.subplots()

    img = plt.imread(im_path)
    ax.imshow(img)

    im_w = img.shape[1]
    im_h = img.shape[0]

    x = int((x - w/2) * im_w)
    y = int((y - h/2) * im_h)
    w = int(w * im_w)
    h = int(h * im_h)

    ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))

    plt.show()


def main():
    prepare_data_for_yolo_training()
    # show_original_image_with_plate('data/origin/openalpr/endtoend/eu/eu4.jpg')
    # show_train_image_with_plate('data/yolo/data/eu4.jpg')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# darknet detector train data/yolo/yolo.data data/yolo/yolo.cfg data/yolo/darknet53.conv.74
# darknet detector train data/yolo/yolo.data data/yolo/yolo.cfg data/yolo/backup/yolo.backup -clear
