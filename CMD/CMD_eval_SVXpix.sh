## 500
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 3 --n_sv 200 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9764430866853625, 'NSV': 562.0645161290323}

## 1000
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 4 --n_sv 300 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9807958574172815, 'NSV': 1161.3225806451612}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 3 --n_sv 400 --unfold 10
# Evaluate_Summary: Epoch [-01/3500], {'ASA': 0.9782949946033762, 'NSV': 1091.3225806451612}

## 1500
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 5 --n_sv 300 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9825587645662268, 'NSV': 1476.8064516129032}

## 2000
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 200 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9843722012733783, 'NSV': 2161.7419354838707}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 7 --n_sv 300 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9843782346062325, 'NSV': 2125.0967741935483}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 6 --n_sv 350 --unfold 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9841728361617892, 'NSV': 2149.7419354838707}

## debug option --nSliceEval
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 200 --unfold 10 --nSliceEval 1
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9843728642018065, 'NSV': 2161.6774193548385}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 200 --unfold 10 --nSliceEval 2
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.983691392885993, 'NSV': 1963.0}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 200 --unfold 10 --nSliceEval 6
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9816964225737961, 'NSV': 1759.3548387096773}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 200 --unfold 10 --nSliceEval 10
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.976363844812069, 'NSV': 1502.7096774193549}

## hier + SVX_pix
## 1000 * 0.5
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 5 --n_sv 200 --unfold 10 --hier_ratio 0.5
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9780570858906199, 'NSV': 457.8709677419355}
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 4 --n_sv 300 --unfold 10 --hier_ratio 0.5
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9774875009874733, 'NSV': 527.9354838709677}

## 2000 * 0.5
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 7 --n_sv 300 --unfold 10 --hier_ratio 0.5
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.9820135404770376, 'NSV': 1006.4516129032259}

## 3000 * 0.5
# CUDA_VISIBLE_DEVICES=0 python ../main_svx_davis.py --evalMode --no_cnn --t_sv 10 --n_sv 300 --unfold 10 --hier_ratio 0.5
# Evaluate_Summary: Epoch [-01/3000], {'ASA': 0.983931214752935, 'NSV': 1512.5483870967741}