
import config
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import numpy as np

# image is assumed to be a single PIL Image
def run_ssd(model, image, min_score=config.MIN_SCORE_2DETECTION,
                      max_iou=config.MAX_IOU,
                      top_k=config.TOP_K,
                      image_transform=None):

    label_list = config.CLASSES

    if image_transform is None:
        image_transform = config.IMAGE_TRANFORM

    x = image_transform(image)
    x = x.unsqueeze(0)
    x = x.to(model.device)

    with torch.no_grad():
        model(x)
        boxes, labels, scores = model.detect(min_score, max_iou, top_k)

    w, h = image.size
    boxes = boxes[0].cpu().numpy()
    boxes[:, 0:4:2] = boxes[:, 0:4:2] * w
    boxes[:, 1:4:2] = boxes[:, 1:4:2] * h

    labels = labels[0].cpu().numpy()
    scores = scores[0].cpu().numpy()

    image2 = image.copy()
    canvas = ImageDraw.Draw(image2)
    for box, label, score in zip(boxes, labels, scores):
        if label != 0:
            lefttop = (int(box[0] - box[2] // 2), int(box[1] - box[3] // 2))
            rightbottom = (int(box[0] + box[2] // 2), int(box[1] + box[3] // 2))
            canvas.rectangle([lefttop, rightbottom], outline="red")
            canvas.text((lefttop[0], rightbottom[1]-10), label_list[label-1]+': {:.3f}'.format(score), fill=(255, 0, 0, 255))
            print(label_list[label-1]+': {:.3f}'.format(score))
    # plt.imshow(image2)
    # plt.show()
    image2.show()
    return boxes, labels, scores

if __name__ == '__main__':

    import argparse
    import model
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description='SSD inference')
    parser.add_argument('--weights', type=str, help='path to model.pth', required=True)
    parser.add_argument('--image', type=str, help='path to image', required=True)
    parser.add_argument('--min_score', type=float, help='minimum score', default=config.MIN_SCORE_2DETECTION)
    parser.add_argument('--max_iou', type=float, default=config.MAX_IOU)
    parser.add_argument('--top_k', type=int, default=config.TOP_K)
    args = parser.parse_args()

    net = model.MobileNetV2_SSD().to(config.DEVICE)
    net.load_state_dict(torch.load(args.weights, map_location=config.DEVICE))
    net.eval()
    image = Image.open(args.image).convert("RGB")
    run_ssd(net, image, args.min_score, args.max_iou, args.top_k)
