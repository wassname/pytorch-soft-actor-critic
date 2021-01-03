run:
	python main.py --demonstrations data/demonstrations  --tau 1 --target_update_interval 100

play:
	python play.py --load-actor models/actor_2021-01-02_10-26-23_SAC_ApplePick-v0_Gaussian_autotune.pkl --load-critic models/critic_2021-01-02_10-26-23_SAC_ApplePick-v0_Gaussian_autotune.pkl --render
