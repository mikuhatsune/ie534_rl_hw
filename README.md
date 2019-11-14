## IE 534 Assignment: Reinforcement Learning

#### Getting Started
You can either:

* ~~Fork your own copy of the repo, and work on it~~
  Just realized this is probably not a good idea, because you can see other people's solutions...
Try to do a private fork then: https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private

* or, download a zip file containing everything: https://github.com/mikuhatsune/ie534_rl_hw/archive/master.zip (**recommended**)
* or, directly clone the repo to local:
```bash
git clone https://github.com/mikuhatsune/ie534_rl_hw.git
```

Please follow instructions in the Jupyter notebook [rl.ipynb](rl.ipynb).

An example of finished homework is in [example_solution/rl.ipynb](example_solution/rl.ipynb) and [example_solution/rl.pdf](example_solution/rl.pdf).

Example training logs [example_solution/log_breakout_dqn.txt](example_solution/log_breakout_dqn.txt), and [example_solution/log_breakout_a2c.txt](example_solution/log_breakout_a2c.txt).
Format:
```
iter: iteration
n_ep: number of episodes (games played)
ep_len: running averaged episode length
ep_rew: running averaged episode clipped reward
raw_ep_rew: running averaged raw episode reward (actual raw game score)
env_step: number of environment simulation steps
time, rem: time passed, estimated time remain

iter    500 |loss   0.00 |n_ep    28 |ep_len   31.3 |ep_rew  -0.22 |raw_ep_rew   1.76 |env_step   1000 |time 00:04 rem 281:49
```

#### Important
Run these commands once to make BlueWaters happy (install a newer version of gym):
```bash
module load python/2.0.0
pip install gym[atari]==0.14 --user
```
