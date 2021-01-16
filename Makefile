python=/home/wassname/anaconda/envs/diygym3/bin/python
date=2021-01-03_13-30-07
LOGURU_LEVEL=INFO
run:
	LOGURU_LEVEL=INFO ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 4 --automatic_entropy_tuning
	# LOGURU_LEVEL=INFO ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2 --load auto --alpha 0.1 --tau 1 --target_update_interval 1000
	# LOGURU_LEVEL=INFO ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2 --load auto --tau 1 --target_update_interval 1000 --policy Deterministic
	
	# ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 2 --load

	# LOGURU_LEVEL=INFO ${python} main.py --demonstrations data/demonstrations --cuda --updates_per_step 4 --load auto --automatic_entropy_tuning

	# hard update
	
	#models/2021-01-10_12-49-47_SAC_ApplePick-v0_Gaussian_autotune0

play:
	# ${python} play.py --load-actor models/actor_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --load-critic models/critic_${date}_SAC_ApplePick-v0_Gaussian_autotune.pkl --render
	${python} main.py --load auto --render --num_steps 0 --no-train

