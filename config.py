
import torch
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------
# Dataset/Dataloader Configuration
# --------------------------------
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

IMAGE_SIZE = 256
IMAGE_TRANFORM = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                     transforms.CenterCrop(IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
# max annotations per image
MAX_N_BOX = 5

# -------------------
# Model Configuration
# -------------------
# affects build
# use a pretrained mobilenetv2 for backbone
MOBILENET_PRETRAINED = True
# number of boxes generated from each latent space
N_BOX = 1
# offset for the generated boxes
BOX_PRIOR = [[0.5, 0.5, 0.6, 0.6]]
# class prior
#CLASS_PRIOR = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
CLASS_PRIOR = [1. / (1 + len(CLASSES))] * (1 + len(CLASSES))

# affects training
# number of negatives that are used for training
NEGATIVE_POSTIVE_RATIO = 3.
# coefficient for localization loss
ALPHA = 1.
# minimum iou for the generated box to qualify as positive
MIN_IOU_2POSTIVE = 0.8

# affects inference
# minimum confidence for detection
MIN_SCORE_2DETECTION = 0.5
# maximum iou boxes can have without suppression
MAX_IOU = 0.5
# only show top k boxes
TOP_K = 1



# ----------------------
# Training Configuration
# ----------------------
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
EPOCHS = 50
