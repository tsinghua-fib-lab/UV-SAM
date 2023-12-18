import os
import sys

sys.path.insert(0, sys.path[0]+'/.')


import mmseg
import numpy as np
import cv2

from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random

import os
import numpy as np
from mmseg.apis import init_model, inference_model
import torch
import imgviz
from scipy.ndimage import binary_dilation,binary_erosion
import mmcv
import mmengine
import numpy as np
from mmengine import Config, get
from mmengine.dataset import Compose
from mmpl.registry import MODELS, VISUALIZERS
from mmpl.utils import register_all_modules
register_all_modules()
import argparse
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)
from glob import glob

def find_contours(sub_mask):
    _, thresh = cv2.threshold(sub_mask, 0, 255, cv2.THRESH_BINARY)

    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
import cv2
def fill_hollow_area(mask):

    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Find contours of the edges
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours
    for contour in contours:
        # Check if any pixel in the contour is zero (indicating a hollow area)
        contour_mask = np.zeros_like(binary_mask)
        
        # Mask the grayscale image with the contour mask
        masked_image = cv2.bitwise_and(binary_mask, binary_mask, mask=contour_mask)
    
        # Check if any pixel in the masked image is zero (indicating a hollow area)
        if np.any(masked_image == 0):
        
            # Perform flood fill on the hollow area
            cv2.drawContours(mask, [contour], 0, 1, -1)
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # Find contours of the edges
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate over the contours
    for contour in contours:
        if abs(cv2.contourArea(contour))<100*100 or cv2.boundingRect(contour)[2] < 100 or cv2.boundingRect(contour)[3] < 100:
            cv2.drawContours(mask, [contour], 0, 0, -1)   
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Find contours of the edges
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over the contours
    for contour in contours:
        # Check if any pixel in the contour is zero (indicating a hollow area)
        contour_mask = np.zeros_like(binary_mask)
        
        # Mask the grayscale image with the contour mask
        masked_image = cv2.bitwise_and(binary_mask, binary_mask, mask=contour_mask)
    
        # Check if any pixel in the masked image is zero (indicating a hollow area)
        if np.any(masked_image == 0):
        
            # Perform flood fill on the hollow area
            cv2.drawContours(mask, [contour], 0, 1, -1)


    
    return mask
def calculate_ac(true_positive, false_positive,false_negatives):
    precision = true_positive / (true_positive + false_positive)
    recall=true_positive / (true_positive + false_negatives)
    f1_score=2*precision*recall/(precision+recall)
    return precision,recall,f1_score

def construct_sample(img, pipeline):
    img = np.array(img)[:, :, ::-1]
    inputs = {
        'ori_shape': img.shape[:2],
        'img': img,
    }
    pipeline = Compose(pipeline)
    sample = pipeline(inputs)
    return sample
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'

def build_model(cp, model_cfg):
    model_cpkt = torch.load(cp, map_location='cpu')
    model = MODELS.build(model_cfg)
    model.load_state_dict(model_cpkt['state_dict'], strict=True)
    model.to(device=device)
    model.eval()
    return model




def cal_pr_f1(true_masks,pred_masks):
        
    pred_contours=find_contours(pred_masks)
    true_contours=find_contours(true_masks)
    iou_matrix = np.zeros((len(pred_contours), len(true_contours)))
    for idx in range(len(pred_contours)):
        for jdx in range(len(true_contours)):
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            contour_pred_mask=cv2.drawContours(mask, pred_contours,idx, color=1, thickness=cv2.FILLED)
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            contour_true_mask=cv2.drawContours(mask, true_contours,jdx, color=1, thickness=cv2.FILLED)

            intersection = np.logical_and(contour_pred_mask, contour_true_mask)
            union = np.logical_or(contour_pred_mask, contour_true_mask)
            iou = np.sum(intersection) / np.sum(union)
            iou_matrix[idx, jdx] = iou
    threshold = 0.0 
    true_positives_cnt=0
    false_positives_cnt=0
    for i in range(len(pred_contours)):
        best_match = np.argmax(iou_matrix[i])
        if iou_matrix[i, best_match] > threshold:
            true_positives_cnt+=1
            iou_matrix[:, best_match] = 0
        else:
            false_positives_cnt+=1
    false_negatives_cnt = len(true_contours)- true_positives_cnt 
    return true_positives_cnt,false_positives_cnt,false_negatives_cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="/configs/rsprompter/samseg_segformer_bbox_emb_MLP_SAMval_config.py",help='train config file path')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.0001)    
    parser.add_argument('--dilation_iter', type=int, default=8)
    args = parser.parse_args()



    checkpoint = '/checkpoints/xxxx/xxxx.ckpt'#put your model path here
    cfg = Config.fromfile(args.cfg)

    model = build_model(checkpoint, cfg.model_cfg)




    method=os.path.basename(args.config).split("/")[-1].split(".")[0]


    gt_folder="/data/mask/test"


    im_folder="/data/image/test/"

    if not os.path.exists("/results_inference/mask_outputs/"+method+"_"+str(args.lr)+"_"+str(args.wd)+"/"):
        os.makedirs("/results_inference/mask_outputs/"+method+"_"+str(args.lr)+"_"+str(args.wd)+"/")

    model.eval()
    sam_iou=[]
    mask_rcnn_iou=[]
    sam_precision=[]
    sam_recall=[]
    sam_f1=[]
    seg_precision=[]
    seg_recall=[]
    seg_f1=[]
    score_list=[]
    for dilation_iter in range(4,6,4):
        args.dilation_iter=dilation_iter
        score_list.append(dilation_iter)



        seg_root_dir="/results_inference/"+method+"_"+str(args.lr)+"_"+str(args.wd)+"/"
        if not os.path.exists(seg_root_dir):
            os.mkdir(seg_root_dir)
            


        from tqdm import tqdm
        cnt_idx=0
        mask_rcnn_bdata= np.zeros([len(os.listdir(im_folder)), 1024, 1024, 2]).astype(bool)
        SAM_bdata= np.zeros([len(os.listdir(im_folder)), 1024, 1024, 2]).astype(bool)

        seg_tp=0
        seg_fp=0
        seg_fn=0
        sam_tp=0
        sam_fp=0
        sam_fn=0

        for im_file in tqdm(sorted(os.listdir(im_folder))):

            im = mmcv.imread(os.path.join(im_folder, im_file))
            sample = construct_sample(im, cfg.predict_pipeline)
            sample['inputs'] = [sample['inputs']]
            sample['data_samples'] = [sample['data_samples']]



            
            ground_truth = Image.open(gt_folder+im_file)
                
            ground_truth = np.array(ground_truth)

            height = im.shape[0]
            width = im.shape[1]

            one_image_mask=np.zeros((1024,1024),dtype=np.uint8)

            with torch.no_grad():
                outputs = model.predict_step(sample, batch_idx=0)
            masks=np.array(outputs[0].pred_sem_seg.data.cpu().numpy(), dtype=np.uint8)


            pred_masks =  masks.reshape(1024,1024)
            pred_boxes=[]
            pred_contours = find_contours(pred_masks)

            one_image_mask=np.zeros((1024,1024),dtype=np.uint8)

            for idx,contour in enumerate(pred_contours):
                contour_mask = np.zeros_like(pred_masks)

                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                contour_box=cv2.boundingRect(contour)
                contour_box=[contour_box[0],contour_box[1],contour_box[0]+contour_box[2],contour_box[1]+contour_box[3]]
                if abs(cv2.contourArea(contour)) > 100*100 and  contour_box[2]>100 and contour_box[3]>100:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    one_image_mask+=cv2.drawContours(mask, pred_contours,idx, color=1, thickness=cv2.FILLED)
                    pred_boxes.append(contour_box)
            one_image_mask[one_image_mask>1]=1

            temp=cal_pr_f1(ground_truth,one_image_mask)      
            seg_tp+=temp[0]
            seg_fp+=temp[1]
            seg_fn+=temp[2]
            

            save_colored_mask(one_image_mask.reshape(1024,1024), seg_root_dir+im_file)
            
            mask_rcnn_pred=one_image_mask.reshape(1024,1024).copy()

            mask_rcnn_bdata[cnt_idx, :, :, 0] = ground_truth == 1
            mask_rcnn_bdata[cnt_idx, :, :, 1] = mask_rcnn_pred == 1 

            cnt_idx+=1
        mask_rcnn_iou.append(np.sum(mask_rcnn_bdata[..., 0] & mask_rcnn_bdata[..., 1])/np.sum(mask_rcnn_bdata[..., 0] | mask_rcnn_bdata[..., 1])) #计算IOU
        temp=calculate_ac(seg_tp,seg_fp,seg_fn)
        seg_precision.append(temp[0])
        seg_recall.append(temp[1])
        seg_f1.append(temp[2])
        print("seg_iou:",mask_rcnn_iou)



import csv


final_list=[]
final_list.append(score_list)

final_list.append(mask_rcnn_iou)
final_list.append(seg_precision)

final_list.append(seg_recall)
final_list.append(seg_f1)


if not os.path.exists("/results_inference/"):
    os.makedirs("/results_inference/")

with open("/results_inference/"+method+"_"+str(args.lr)+"_"+str(args.wd)+".csv", 'w') as cf:
    csvfile = csv.writer(cf, delimiter=',')
    for column in zip(*[i for i in final_list]):
        csvfile.writerow(column)
cf.close() 

