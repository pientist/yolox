CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 \
python3 tools/train.py \
--exp_file exps/example/custom/yolox_s.py \
--devices 6 \
--batch_size 24 \
--fp16 \
--ckpt ./yolox_s.pth