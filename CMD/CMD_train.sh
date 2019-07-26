CUDA_VISIBLE_DEVICES=0,1 python ../main_svx_davis16.py --evalStep 100000 \
--crop_size 16 201 201 \
--p_scale 0.25 --lab_scale 0.26 --t_sv 3 --n_sv 100 \
--unfold 5 --softscale -1.0 \
--w_pos 0.1 --w_col 0 --w_label 10 \
--saveDir 'results/exp1/' --nEpoch 1000 --checkpoint 800
