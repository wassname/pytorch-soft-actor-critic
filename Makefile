python=/home/wassname/anaconda/envs/diy-gym2/bin/python
date=2021-01-03_13-30-07
run:
	${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2

play:
	${python} play.py --load-actor models/actor_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --load-critic models/critic_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --render

