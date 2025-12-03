torchrun --nproc-per-node 4 dinov3/train/train.py \
--output-dir /home/digipath2/projects/foundation_dinov3/out/convnext_tiny/ \
--config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
train.dataset_path=DigipathPatches:root=/home/digipath2/projects/foundation/dataset_foundation_2048_5x.dt \
student.arch=convnext_tiny \
crops.global_crops_size=1024 \
crops.local_crops_size=448 \
crops.global_crops_scale=[0.225,0.275] \
crops.local_crops_scale=[0.0431,0.0526] \
crops.rgb_mean=[0.0,0.0,0.0] \
crops.rgb_std=[1.0,1.0,1.0] \
student.patch_size=32 \
train.batch_size_per_gpu=8