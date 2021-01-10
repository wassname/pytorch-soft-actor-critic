python=/home/wassname/anaconda/envs/diygym3/bin/python
date=2021-01-03_13-30-07
LOGURU_LEVEL=INFO
run:
	# ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2 --load
	LOGURU_LEVEL=INFO ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2 --load models/2021-01-09_21-34-39_SAC_ApplePick-v0_Gaussian_autotune

play:
	# ${python} play.py --load-actor models/actor_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --load-critic models/critic_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --render
	${python} main.py --load auto --render --num_steps 0 --no-train

