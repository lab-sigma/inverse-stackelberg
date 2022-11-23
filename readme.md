### Inverse Game Theory for Stackelberg Games: the Blessing of Bounded Rationality

```
### To generate game instances
python game.py

### To run experiments 

# learn follower utility of a Stackelberg game m=n=10x10 alpha=0.2 using PURE
python run.py --T 10000000 --realized --game_name 10x10-0.2 --exp_lambda 8 --method_name pure

# learn follower utility of a Stackelberg game m=n=10x10 alpha=0.2 using PURE-EXP
python run.py --T 10000000 --realized --game_name 10x10-0.2 --exp_lambda 8 --method_name pure_exp

# learn follower utility in security game with 20 targets using PURE with structure insight
python run.py --T 10000000 --realized --game_name sec-20 --exp_lambda 8 --method_name pure_sec


```