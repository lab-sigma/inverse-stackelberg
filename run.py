import argparse, os, math, random

from logger import *
from game import *
from pure import pure
from pure_exp import pure_exp
from pure_sec import pure_sec

parser = argparse.ArgumentParser()
describe = lambda names :  ''.join( [', {}: {}'.format(i, n) for i,n in enumerate(names)] )

parser.add_argument('--T', type=int, default=10000000, help='T')
parser.add_argument('--realized', default=False, action='store_true', help='whether to use true_loss')
parser.add_argument('--game_name', type=str, default='10x10-0.2', help='')
parser.add_argument('--exp_lambda', type=int, default=8, help='lambda')
parser.add_argument('--method_name', type=str, default='pure', help='pure, pure_exp, pure_sec')
parser.add_argument('--update_period', type=int, default=100000, help='number of round between each log checkpoint')

args = parser.parse_args()

game_dir = "game-profile/"
log_dir = "results/"


dirname = f"{log_dir}{args.game_name}/{args.method_name}/"
if not os.path.exists(dirname):
    os.makedirs(dirname)
for n in range(1,2):
    game_filename = f"{game_dir}{args.game_name}#{n}" 
    game = load_game(game_filename)
    log_filename = f"{dirname}{args.exp_lambda}#{n}"
    if os.path.exists(log_filename+".png"): 
        print("skipping for ", log_filename+".png")
        continue
    lg = logger(log_filename, game, args)
    if args.method_name == "pure":
        pure(game, lg, args.exp_lambda, args)
    elif args.method_name == "pure_exp":
        pure_exp(game, lg, args.exp_lambda, args)
    elif args.method_name == "pure_sec":
        pure_sec(game, lg, args.exp_lambda, args)
    else:
        print("method_name not found")
        exit(1)

    lg.plot()