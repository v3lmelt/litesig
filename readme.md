# LiteSig: A Parameter-Efficient Hybrid ViT-CNN Model for Offline Handwritten Signature Verification

LiteSig is a hybrid Siamese network for offline handwritten signature verification. It combines a lightweight local CNN branch with a DINOv3 ViT-S/16 global branch, then learns pairwise similarity for writer-disjoint verification.

## Prerequisites

- Python 3.10 or later is recommended.
- Install PyTorch and torchvision for your CUDA or CPU environment by following the official PyTorch installation guide.
- Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

- Apply for access to DINOv3 at https://github.com/facebookresearch/dinov3. This project uses the DINOv3 ViT-S/16 distilled checkpoint. With the default `dino_vit` backbone, keep the DINOv3 repository available as `./dinov3` for `torch.hub`, and either place the checkpoint file in the project root or pass its path with `--dino-ckpt`.
- Download and prepare one of the supported signature datasets listed below.

## Dataset

Datasets

- CEDAR and BHSig260: https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets
- UTSIG: https://www.kaggle.com/datasets/sinjinir1999/utsignature-verification
- GPDSSynthetic: requires approval from the dataset provider. Please see reference about GPDS in the paper.

## Training

Run the training script with:
```Python
python train/train.py --help
```
Detailed argument explanation is included. 

Example command:

```bash
python train/train.py \
  --data-root dataset/BHSig160 \
  --dino-ckpt dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --epochs 35 \
  --batch-size 128 \
  --base-lr 5e-5 \
  --backbone-lr-mult 0.1 \
  --weight-decay 1e-2 \
  --head-weight-decay-mult 0.5 \
  --dropout 0.20 \
  --loss-type bce_logits \
  --bce-pos-weight 1.0 \
  --stage1-freeze-epochs 5 \
  --warmup-epochs 2.0 \
  --min-lr-ratio 0.01 \
  --early-stopping-patience 10 \
  --writer-split-ratio 0.625 \
  --train-positive-pairs 100 \
  --train-negative-pairs 100 \
  --val-positive-pairs 274 \
  --val-negative-pairs 274 \
  --resample-train-pairs-each-epoch \
  --img-height 224 \
  --img-width 224 \
  --mixed-precision \
  --amp-dtype fp16 \
  --num-workers 32 \
  --pin-memory
```