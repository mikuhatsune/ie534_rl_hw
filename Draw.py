# Save some images...

import torch
import Env, Algo
import numpy as np
import argparse, os
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=150, help='# testing iterations')
parser.add_argument('--env', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'a2c'])
parser.add_argument('--frame_skip', type=int, default=1, help='for Atari, skip frames by repeating action, useful for NoFrameSkip-v4 envionments')
parser.add_argument('--frame_stack', type=int, default=4, help='for Atari, stack 4 consecutive frames together as observations')
parser.add_argument('--save_dir', type=str, default='breakout_imgs/')
parser.add_argument('--load', type=str, default='breakout/2000000.pth')

args = parser.parse_args()

if args.save_dir[-1] != '/':
    args.save_dir += '/'
try: os.makedirs(args.save_dir)
except: pass

env = Env.LocalEnv(args.env, 1, args.frame_skip, args.frame_stack, save_img_dir=args.save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ('running on device', device)

if args.algo == 'dqn':
	# force epsilon greedy == 0.01
    algo = Algo.DQN(env.obs_space, env.act_space, eps_decay=1, device=device)
elif args.algo == 'a2c':
    algo = Algo.ActorCritic(env.obs_space, env.act_space, device=device)

if len(args.load):
    algo.load(args.load)
    print ("load pretrained model from", args.load)

obses = env.reset()

for it in range(args.niter):
    actions = algo.act(obses)
    results = env.step(actions)
    obses = next(zip(*results))

env.close()
