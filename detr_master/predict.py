import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
# print('aaaa')
from new_models import build_model
# print('bbbb')
from PIL import Image
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Predict with trained DETR model', add_help=False)
    parser.add_argument('--device', default='cuda', help='Device to use for inference')
    parser.add_argument('--model_path', default='checkpoint0299.pth', type=str, help='Path to the trained .pth model file')
    parser.add_argument('--input_image', default='frame_000000.PNG', type=str, help='Path to the input image')
    parser.add_argument('--output_dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--confidence_threshold', default=0.9, type=float, help='Threshold on the confidence of detection')
    return parser

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    bboxes = out_bbox * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return bboxes.numpy()

def predict_main(args,model_path,input_image):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Configure model
    model_args = torch.load(model_path)['args']
    model, _, postprocessors = build_model(model_args)
    model.to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Image preprocessing
    transform = Compose([
        Resize(800),  # Resize image to match the model's expected input
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform image
    # Convert image object to tensor directly if it's not a file path
    if isinstance(input_image, Image.Image):
        img = input_image.convert("RGB")
    else:
        # If input_image is a file path
        img = Image.open(input_image).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Process outputs (e.g., thresholding)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > args.confidence_threshold
    scores = probas.max(-1).values[keep].cpu().numpy()
    classes = probas.argmax(-1)[keep].cpu().numpy()

    # Convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), img.size)

    # Store results in a dictionary
    results = []
    for score, clss, bbox in zip(scores, classes, bboxes_scaled):
        result = {
            "class_id": clss,
            "score": score,
            "bbox": bbox.tolist()  # Convert numpy array to list
        }
        results.append(result)

    # Optionally print the results
    for res in results:
        # print(f"Class: {res['class_id']}, Score: {res['score']:.2f}, Bbox: {res['bbox']}")
        pass

    # Return the results as a list of dictionaries
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    predict_main(args,'checkpoint0299.pth','frame_000000.PNG')

# import torch
# from torchvision.transforms import Compose, ToTensor, Normalize, Resize
# from new_models import build_model
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import argparse
#
# def get_args_parser():
#     parser = argparse.ArgumentParser('Predict with trained DETR model', add_help=False)
#     parser.add_argument('--device', default='cuda', help='Device to use for inference')
#     parser.add_argument('--model_path', default='checkpoint0299.pth', type=str, help='Path to the trained .pth model file')
#     parser.add_argument('--input_image', default='frame_000000.PNG', type=str, help='Path to the input image')
#     parser.add_argument('--output_dir', default='outputs', help='Directory to save outputs')
#     parser.add_argument('--confidence_threshold', default=0.9, type=float, help='Threshold on the confidence of detection')
#     return parser
#
# def rescale_bboxes(out_bbox, size):
#     img_w, img_h = size
#     scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
#     bboxes = out_bbox * scale
#
#     # 确保坐标顺序为（xmin, ymin, xmax, ymax）
#     bboxes[:, 2] = torch.max(bboxes[:, 0], bboxes[:, 2])
#     bboxes[:, 3] = torch.max(bboxes[:, 1], bboxes[:, 3])
#     bboxes[:, 0] = torch.min(bboxes[:, 0], bboxes[:, 2])
#     bboxes[:, 1] = torch.min(bboxes[:, 1], bboxes[:, 3])
#
#     return bboxes.numpy()
#
#
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     # Configure model
#     model_args = torch.load(args.model_path)['args']
#     model, _, postprocessors = build_model(model_args)
#     model.to(device)
#
#     # Load model weights
#     checkpoint = torch.load(args.model_path, map_location=device)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#
#     # Image preprocessing
#     transform = Compose([
#         Resize(800),  # Resize image to match the model's expected input
#         ToTensor(),
#         Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # Load and transform image
#     img = Image.open(args.input_image).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0).to(device)
#
#     # Model inference
#     with torch.no_grad():
#         outputs = model(img_tensor)
#
#     # Process outputs
#     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#     keep = probas.max(-1).values > args.confidence_threshold
#     scores = probas.max(-1).values[keep].cpu().numpy()
#     classes = probas.argmax(-1)[keep].cpu().numpy()
#
#     # Convert boxes from [0; 1] to image scales
#     bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), img.size)
#
#     # Visualization
#     draw = ImageDraw.Draw(img)
#     for score, clss, bbox in zip(scores, classes, bboxes_scaled):
#         draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline="red", width=3)
#         draw.text((bbox[0], bbox[1]), f"{clss} {score:.2f}", fill="red")
#
#     # Display image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('DETR inference script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     main(args)

