CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
taskset -c 0-48 \
python3 tools/train.py \
--exp_file exps/example/custom/yolox_s.py \
--devices 6 \
--batch_size 24 \
--fp16 \
--ckpt YOLOX_outputs/yolox_s/best_ckpt.pth