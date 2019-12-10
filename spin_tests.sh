#! /bin/bash

# random baseline
python3 test.py 1 1 random 3 250 100 none
python3 test.py 1 2 random 3 250 100 none
python3 test.py 2 1 random 3 250 100 none
python3 test.py 1 1 random 3 1000 100 two_walls
python3 test.py 1 2 random 3 1000 100 two_walls
python3 test.py 2 1 random 3 1000 100 two_walls

# reinforce

python3 test.py 1 1 reinforce 3 250 100 none
python3 test.py 1 2 reinforce 3 250 100 none
python3 test.py 2 1 reinforce 3 250 100 none
python3 test.py 1 1 reinforce 3 1000 100 two_walls
python3 test.py 1 2 reinforce 3 1000 100 two_walls
python3 test.py 2 1 reinforce 3 1000 100 two_walls

# actor_critic

python3 test.py 1 1 actor_critic 3 250 100 none
python3 test.py 1 2 actor_critic 3 250 100 none
python3 test.py 2 1 actor_critic 3 250 100 none
python3 test.py 1 1 actor_critic 3 1000 100 two_walls
python3 test.py 1 2 actor_critic 3 1000 100 two_walls
python3 test.py 2 1 actor_critic 3 1000 100 two_walls

# ppo

python3 test.py 1 1 ppo 3 250 100 none
python3 test.py 1 2 ppo 3 250 100 none
python3 test.py 2 1 ppo 3 250 100 none
python3 test.py 1 1 ppo 3 1000 100 two_walls
python3 test.py 1 2 ppo 3 1000 100 two_walls
python3 test.py 2 1 ppo 3 1000 100 two_walls
