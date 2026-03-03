export TF_CPP_MIN_LOG_LEVEL=2

python main_synthetic.py
# python ablation1.py
# python ablation2.py
# python ablation3.py
# python ablation4.py

python main_lastfm1k.py \
  --data-root data \
  --top-n-tags 100 \
  --top-k-users 10 \
  --trajectories-per-user 75 \
  --horizons 10 20 30 40 50 60 70 80 90 100 \
  --n-repeat 30 \
  --ours-em-iters 10 \
  --ours-laplace 0.1 \
  --mdpmix-em-laplace 0.1 \
  --mdpmix-thresh 1e-6 5e-6 1e-5 \
#   --diag