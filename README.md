# Robust MARL

Based on the repo Recurrent-MADDPG: https://github.com/nicoring/rec-maddpg

Implemented algorithms:
* DDPG
* MADDPG
* M3DDPG
* Robust-MADDPG

-----

Training: `python trainer.py --scenario simple_adversary --exp-run-num 0 --good-policy maddpg --adv-policy maddpg --noise-std 0`
- scenario: {simple_crypto, simple_push, simple_adversary, simple_tag, simple_spread}
- exp-run-num: the ith trial, default 0
- good-policy, adv-policy: {ddpg, maddpg, m3ddpg, rmaddpg}
- noise-std: uncertainty level of rewards {0.0 (default, the true rewards), 0.5, 1, 1.5, 2, 2.5, 3 ...}

Visualization: `python trainer.py --scenario simple_adversary --eval --render --good-policy m3ddpg --adv-policy m3ddpg --resume results/simple_adversary_False/good_m3ddpg_adv_m3ddpg_noise_0.00_num_0_2020-05-25T05:29:16/`
- Load model from the given directory, visualize 10 episodes

Evaluation: `python trainer.py --scenario simple_adversary --eval --good-policy m3ddpg --adv-policy m3ddpg --resume results/simple_adversary_False/good_m3ddpg_adv_m3ddpg_noise_0.00_num_0_2020-05-25T05:29:16/`
- Evaluate the trained model for 1000 episodes, report mean accumulated rewards for each agent

Benchmark: `python trainer.py --scenario simple_adversary --benchmark --good-policy m3ddpg --adv-policy m3ddpg --resume results/simple_adversary_False/good_m3ddpg_adv_m3ddpg_noise_0.00_num_0_2020-05-25T05:29:16/`
- Collect benchmark data for producing tables and figures.
