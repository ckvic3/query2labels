CUDA_VISIBLE_DEVICES=0,2,3  python -m torch.distributed.launch --nproc_per_node=3  train.py --world-size 3 --batch-size 48