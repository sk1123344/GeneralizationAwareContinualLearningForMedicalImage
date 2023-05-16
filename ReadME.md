# GeneralizationAbilityAwareContinualLearningMethod for Medical Image

This is the repository for a Continual Learning(CL) method

# Method

## Margin Loss

Use cosine as the measure in the margin loss, and it aware the sample nums for each class. See ``CosineLabelAwareMarginLoss`` in 
`Loss.py`. The method contains [cosface](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1797.pdf) and [LabelDistributionAwareMarginLoss](https://arxiv.org/pdf/1906.07413.pdf)

## Decoder

Use a Transformer Decoder to replace the pooling and fully connected layers in the network(in this case the ResNet18).
See ``TransformerDecoderFundus`` in ``Transformer.py``.

## Contrastive Loss

Use a Contrastive Loss to accelerate the process of adapting to the new data. The replay data doing this can be seen as 
knowledge distillation. And the pseudo feature sampled from each class's distribution can do the similar thing to the 
[Prototype Augmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)

This part is implemented in ``GeneralizationAwareContinualLearning.py``, inside the `_compute_loss()` function.

# Usage

Run ``main.py``, there are several settings can be changed directly in `main.py`.

* dataset_order = [0, 1, 2], 0-ODIR, 1-RR, 2-ISEE
* debug = False, IF TRUE, then only run 1 epoch with several iters, the evaluation may raise error since the prediction may not include all labels
* tasks = 3, maximum tasks, if x only the first x in dataset_order will be used
* load_sub_exp_ckpt = False, if there exists a subset experiment for current ones, load the checkpoint to avoid rerun(e.g. 0->1 is the subset of 0->1->2)
* sample_ratio = 0.1, replay sample ratio for each class
* epochs = 120 
* device = 1, no DDP implementation in these codes, if you want, you can modify the code yourself

