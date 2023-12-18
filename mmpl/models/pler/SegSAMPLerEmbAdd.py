import torch
from typing import List, Any

from mmpl.registry import MODELS, TASK_UTILS
from mmseg.utils import SampleList
from .base_pler import BasePLer
import torch.nn.functional as F
from modules.sam import sam_model_registry
from mmengine.structures import PixelData
from torch import Tensor
from mmpl.utils import ConfigType, OptConfigType

from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
import warnings

from .losses import FocalLoss,DiceLoss
import cv2
import numpy as np

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)
    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

def find_contours(sub_mask):
    _, thresh = cv2.threshold(sub_mask, 0, 255, cv2.THRESH_BINARY)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
@MODELS.register_module()
class SegSAMPLerEmbAdd(BasePLer):
    def __init__(self,
                backbone,
                need_train_names=None,
                seg_path_backbone=None,
                seg_path_pretrained=None,
                seg_path_decode_head=None,
                SAM_weights=1.0,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.need_train_names = need_train_names
        self.SAM_weights=SAM_weights

        backbone_type = backbone.pop('type')
        self.backbone = sam_model_registry[backbone_type](**backbone)

        # seg_path init
        if seg_path_pretrained is not None:
            seg_path_backbone.pretrained = seg_path_pretrained
        self.seg_path_backbone = MODELS.build(seg_path_backbone)
        self.seg_path_decode_head = MODELS.build(seg_path_decode_head)
        self.seg_path_backbone.init_weights()
        self.align_corners = self.seg_path_decode_head.align_corners
        self.num_classes = self.seg_path_decode_head.num_classes 
        self.out_channels = self.seg_path_decode_head.out_channels

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.SAM_focal_loss=FocalLoss()
        self.SAM_dice_loss=DiceLoss()
        self.SAM_MSE_loss=torch.nn.MSELoss()

    def _set_my_grad(self, need_train_names: list=[], noneed_train_names: list=[]):
        print(self.named_parameters())
        for name, param in self.named_parameters():
            flag = False
            for need_train_name in need_train_names:
                if need_train_name in name:
                    flag = True
            for noneed_train_name in noneed_train_names:
                if noneed_train_name in name:
                    flag = False
            param.requires_grad_(flag)
            if param.requires_grad:
                print(name)
        not_specific_names = []
        for name, param in self.named_parameters():
            flag_find = False
            for specific_name in need_train_names + noneed_train_names:
                if specific_name in name:
                    flag_find = True
            if not flag_find:
                not_specific_names.append(name)

        if self.local_rank == 0:
            not_specific_names = [x.split('.')[0] for x in not_specific_names]
            not_specific_names = set(not_specific_names)
            print(f"Turning off gradients for names: {noneed_train_names}")
            print(f"Turning on gradients for names: {need_train_names}")
            print(f"Turning off gradients for not specific names: {not_specific_names}")
    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.need_train_names is not None:
            self._set_my_grad(self.need_train_names, noneed_train_names=[])

    def init_weights(self):
        import ipdb; ipdb.set_trace()
        pass

    def train(self, mode=True):
        if self.need_train_names is not None:
            return self._set_train_module(mode, self.need_train_names)
        else:
            super().train(mode)
            return self

    @torch.no_grad()
    def extract_feat(self, batch_inputs):
    
        feat = self.backbone.image_encoder(batch_inputs)
        return feat

    ##segformer-SAM validation
    def validation_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        batch_img_metas = [ data_sample.metainfo for data_sample in batch_data_samples]
       
        last_features= self.extract_feat(batch_inputs)
        seg_path_feats = self.seg_path_backbone(batch_inputs)
        x=self.seg_path_decode_head.get_fusion_conv_feats(seg_path_feats)

        layer = torch.nn.AvgPool2d(4, stride=4)

        
        prompt_emb=torch.cat([F.normalize(self.avg_pool(last_features).reshape(last_features.shape[0],-1),dim=1),F.normalize(self.avg_pool(x).reshape(x.shape[0],-1),dim=1)],dim=1)
        
        
        #prompt_emb=torch.cat([F.normalize(self.avg_pool(x).reshape(x.shape[0],-1),dim=1)],dim=1) 
        results = self.seg_path_decode_head.predict(seg_path_feats,batch_img_metas,self.train_cfg)        
        results = self.postprocess_result( results,batch_data_samples)
        masks=torch.zeros((batch_inputs.shape[0],1024,1024),device=batch_inputs.device,dtype=torch.float16)
        for idx in range(len(results)):
            one_mask=np.array(results[idx].pred_sem_seg.data.cpu().numpy(), dtype=np.uint8).reshape(1024,1024)   
            one_mask_logits=results[idx].seg_logits.data[1].reshape(1,1,1024,1024)

            one_mask_logits=layer(one_mask_logits)   
            one_contours = find_contours(one_mask)
            for jdx,contour in enumerate(one_contours):

                contour_mask = np.zeros_like(one_mask)

                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                contour_box=cv2.boundingRect(contour)
                contour_box=[contour_box[0],contour_box[1],contour_box[0]+contour_box[2],contour_box[1]+contour_box[3]]
                mask=torch.zeros((1,1024,1024),device=batch_inputs.device,dtype=torch.float32)
                iou_predictions=torch.tensor([1.],device=batch_inputs.device)
                if abs(cv2.contourArea(contour)) > 100*100 and  contour_box[2]>100 and contour_box[3]>100:

                    sparse_embeddings, dense_embeddings=self.backbone.prompt_encoder(points=None,boxes=torch.tensor(contour_box,device=batch_inputs.device,dtype=torch.float32).reshape(1,4),masks=one_mask_logits)
                    #sparse_embeddings=torch.zeros_like(sparse_embeddings,device=batch_inputs.device,dtype=torch.float32)
                    sparse_embeddings[0]+=prompt_emb[idx][0:256]
                    sparse_embeddings[0]+=prompt_emb[idx][256:512]
                    #sparse_embeddings=self.project_head(sparse_embeddings)
                    
                    
                    low_res_masks, iou_predictions = self.backbone.mask_decoder.forward_batch(
                    image_embeddings=last_features[idx].reshape(1,256,64,64),
                    image_pe= self.backbone.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,)


                    mask = F.interpolate(
                            low_res_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                    ).reshape(1024,1024)

                    mask = mask > 0.0
                    mask =  mask.float()
                    masks[idx]+=mask
                    masks[idx][masks[idx]>1]=1
                one_gt_mask=np.zeros((1024,1024))
                one_gt_mask[contour_box[1]:contour_box[3],contour_box[0]:contour_box[2]]=np.array(batch_data_samples[idx].gt_sem_seg.data.cpu().numpy(), dtype=np.uint8).reshape(1024,1024)[contour_box[1]:contour_box[3],contour_box[0]:contour_box[2]]

                one_gt_mask=torch.tensor(one_gt_mask,device=batch_inputs.device,dtype=torch.float32).reshape(1,one_gt_mask.shape[0],one_gt_mask.shape[1])
                


        results = self.postprocess_result( masks.reshape(batch_inputs.shape[0],1,1024,1024),batch_data_samples)
        self.val_evaluator.update(batch, results)
    
    
    def training_step(self, batch, batch_idx):
        data = self.data_preprocessor(batch, True)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        batch_img_metas = [ data_sample.metainfo for data_sample in batch_data_samples]

        last_features= self.extract_feat(batch_inputs)

        seg_path_feats = self.seg_path_backbone(batch_inputs)
        x=self.seg_path_decode_head.get_fusion_conv_feats(seg_path_feats)


        
        prompt_emb=torch.cat([F.normalize(self.avg_pool(last_features).reshape(last_features.shape[0],-1),dim=1),F.normalize(self.avg_pool(x).reshape(x.shape[0],-1),dim=1)],dim=1)


        losses = dict()        
        x=self.seg_path_decode_head.get_cls_seg_feats(x)

        losses= self.seg_path_decode_head.loss_by_feat(x, batch_data_samples)



        results = self.seg_path_decode_head.predict(seg_path_feats,batch_img_metas,self.train_cfg)        
        results = self.postprocess_result( results,batch_data_samples)
        SAM_focal_loss=0
        SAM_dice_loss=0
        SAM_MSE_loss=0
        layer = torch.nn.AvgPool2d(4, stride=4)
        for idx in range(len(results)):
            one_mask=np.array(results[idx].pred_sem_seg.data.cpu().numpy(), dtype=np.uint8).reshape(1024,1024)   
            one_mask_logits=results[idx].seg_logits.data[1].reshape(1,1,1024,1024)

            one_mask_logits=layer(one_mask_logits)            
            one_contours = find_contours(one_mask)
            for jdx,contour in enumerate(one_contours):
                
                contour_mask = np.zeros_like(one_mask)

                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                contour_box=cv2.boundingRect(contour)
                contour_box=[contour_box[0],contour_box[1],contour_box[0]+contour_box[2],contour_box[1]+contour_box[3]]
                mask=torch.zeros((1,1024,1024),device=batch_inputs.device,dtype=torch.float32)
                iou_predictions=torch.tensor([1.],device=batch_inputs.device)
                if abs(cv2.contourArea(contour)) > 100*100 and  contour_box[2]>100 and contour_box[3]>100:

                    sparse_embeddings, dense_embeddings=self.backbone.prompt_encoder(points=None,boxes=torch.tensor(contour_box,device=batch_inputs.device,dtype=torch.float32).reshape(1,4),masks=one_mask_logits)
                    
                    sparse_embeddings[0]+=prompt_emb[idx][0:256]
                    sparse_embeddings[0]+=prompt_emb[idx][256:512]

                    low_res_masks, iou_predictions = self.backbone.mask_decoder.forward_batch(
                    image_embeddings=last_features[idx].reshape(1,256,64,64),
                    image_pe= self.backbone.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,)


                    mask = F.interpolate(
                            low_res_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                    ).reshape(1,1024,1024)
                one_gt_mask=np.zeros((1024,1024))
                one_gt_mask[contour_box[1]:contour_box[3],contour_box[0]:contour_box[2]]=np.array(batch_data_samples[idx].gt_sem_seg.data.cpu().numpy(), dtype=np.uint8).reshape(1024,1024)[contour_box[1]:contour_box[3],contour_box[0]:contour_box[2]]

                one_gt_mask=torch.tensor(one_gt_mask,device=batch_inputs.device,dtype=torch.float32).reshape(1,one_gt_mask.shape[0],one_gt_mask.shape[1])
                #calculate iou
                batch_iou=calc_iou(mask, one_gt_mask)
                SAM_focal_loss+=self.SAM_focal_loss(mask, one_gt_mask)
                SAM_dice_loss+=self.SAM_dice_loss(mask, one_gt_mask)
                SAM_MSE_loss +=  F.mse_loss(iou_predictions, batch_iou, reduction='sum')
                loss_SAM = (SAM_focal_loss+SAM_dice_loss+SAM_MSE_loss)/(len(one_contours)*len(results))
                losses['loss_SAM']=loss_SAM*self.SAM_weights

        parsed_losses, log_vars = self.parse_losses(losses)

        log_vars = {f'train_{k}': v for k, v in log_vars.items()}
        log_vars['loss'] = parsed_losses

        self.log_dict(log_vars, prog_bar=True)
        return log_vars

    def on_before_optimizer_step(self, optimizer) -> None:
        self.log_grad(module=self.seg_path_backbone)
        self.log_grad(module=self.seg_path_decode_head)

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        data = self.data_preprocessor(batch, False)
        batch_inputs = data['inputs']
        batch_data_samples = data['data_samples']
        batch_img_metas = [ data_sample.metainfo for data_sample in batch_data_samples]
       
        last_features= self.extract_feat(batch_inputs)
        seg_path_feats = self.seg_path_backbone(batch_inputs)
        x=self.seg_path_decode_head.get_fusion_conv_feats(seg_path_feats)

        layer = torch.nn.AvgPool2d(4, stride=4)

        
        prompt_emb=torch.cat([F.normalize(self.avg_pool(last_features).reshape(last_features.shape[0],-1),dim=1),F.normalize(self.avg_pool(x).reshape(x.shape[0],-1),dim=1)],dim=1)
        results = self.seg_path_decode_head.predict(seg_path_feats,batch_img_metas,self.train_cfg)        
        results = self.postprocess_result( results,batch_data_samples)
        masks=torch.zeros((batch_inputs.shape[0],1024,1024),device=batch_inputs.device,dtype=torch.float16)
        for idx in range(len(results)):
            one_mask=np.array(results[idx].pred_sem_seg.data.cpu().numpy(), dtype=np.uint8).reshape(1024,1024)   
            one_mask_logits=results[idx].seg_logits.data[1].reshape(1,1,1024,1024)

            one_mask_logits=layer(one_mask_logits)   
            one_contours = find_contours(one_mask)
            for jdx,contour in enumerate(one_contours):
                contour_mask = np.zeros_like(one_mask)

                cv2.drawContours(contour_mask, [contour], 0, 255, thickness=cv2.FILLED)

                contour_box=cv2.boundingRect(contour)
                contour_box=[contour_box[0],contour_box[1],contour_box[0]+contour_box[2],contour_box[1]+contour_box[3]]
                mask=torch.zeros((1,1024,1024),device=batch_inputs.device,dtype=torch.float32)

                if abs(cv2.contourArea(contour)) > 100*100 and  contour_box[2]>100 and contour_box[3]>100:

                    sparse_embeddings, dense_embeddings=self.backbone.prompt_encoder(points=None,boxes=torch.tensor(contour_box,device=batch_inputs.device,dtype=torch.float32).reshape(1,4),masks=one_mask_logits)
                    sparse_embeddings[0]+=prompt_emb[idx][0:256]
                    sparse_embeddings[0]+=prompt_emb[idx][256:512]                   
                    
                    low_res_masks, iou_predictions = self.backbone.mask_decoder.forward_batch(
                    image_embeddings=last_features[idx].reshape(1,256,64,64),
                    image_pe= self.backbone.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,)


                    mask = F.interpolate(
                            low_res_masks,
                            (1024, 1024),
                            mode="bilinear",
                            align_corners=False,
                    ).reshape(1024,1024)
                    mask = mask > 0.0
                    mask =  mask.float()
                    masks[idx]+=mask
                    masks[idx][masks[idx]>1]=1
  


        results = self.postprocess_result( masks.reshape(batch_inputs.shape[0],1,1024,1024),batch_data_samples)
        return results

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_pred = (i_seg_logits >
                              0).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples





