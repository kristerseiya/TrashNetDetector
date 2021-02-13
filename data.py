
import config
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import zipfile
import csv
from PIL import Image

def extractTrashNetZipFile(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall()

def recomputeBox(x_lt, y_lt, box_width, box_height, image_width, image_height):
    x_center =  (box_width / 2. + x_lt) / image_width
    y_center =  (box_height / 2. + y_lt) / image_height
    width = box_width / float(image_width)
    height = box_height / float(image_height)
    return [x_center, y_center, width, height]

# def recomputeBox(format1, format2, x1, x2, x3, x4):
#     if format1 == 'xywh':
#         box = [x1, x2, x3, x4]
#     elif format1 == 'cxcywh':
#         box = [x1-x3//2, x2-x4//2, x3, x4]
#     elif format1 == 'xyxy':
#         box = [x1, x2, x3-x1, x4-x2]
#     else:
#         raise ValueError('sheeeit')
#
#     if format2 == 'xywh':
#         return box
#     elif format2 == 'cxcywh':
#         box = [x1+x3//2, x2+x4//2, x3, x4]
#         return box
#     elif format2 == 'xyxy':
#         box = [x1, x2, x1+x3, x2+x4]
#         return box
#     else:
#         raise ValueError('sheeeit')


class TrashNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        class_set = {cl: idx for cl, idx in zip(config.CLASSES, range(1, len(config.CLASSES) + 1)) }
        self.classes = class_set
        self.images = []
        self.labels = []
        self.boxes = []
        self.max_box = config.MAX_N_BOX

        data = []
        with open(root_dir+'/annotations.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(row)

        data.sort(key=lambda x: x[-3])

        last_image = ''
        labels_per_image = list(self.max_box * [0])
        boxes_per_image = list(self.max_box * [0.5, 0.5, 0., 0.])
        count = 0

        for d in data:

            class_name = d[0]
            box = d[1:5]
            img_path = d[5]
            wh = d[6:]

            if class_name in class_set:
                if img_path != last_image:

                    self.labels.append(list(self.max_box * [0]))
                    self.boxes.append(list(self.max_box * [[0.5, 0.5, 0., 0.]]))

                    # read image
                    temp = Image.open(root_dir+'/'+img_path)
                    keep = temp.copy()
                    temp.close()
                    self.images.append(keep)
                    last_image = img_path

                    # get class label
                    self.labels[-1][0] = class_set[class_name]
                    # get box in [cx, cy, w, h] format
                    box = recomputeBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                                       keep.size[0], keep.size[1])
                    self.boxes[-1][0] = box
                    count = 1

                else:

                    if count >= self.max_box:
                        continue
                    self.labels[-1][count] = class_set[class_name]
                    box = recomputeBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                                       keep.size[0], keep.size[1])
                    self.boxes[-1][count] = box
                    count += 1


        if transform is None:
            self.transform = config.IMAGE_TRANFORM
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        return self.transform(self.images[x]), torch.LongTensor(self.labels[x]), torch.FloatTensor(self.boxes[x])
