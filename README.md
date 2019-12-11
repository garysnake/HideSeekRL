# Multi-Agent Hide and Seek 
### Gary Zhong and Aaron Chang

### Requirements
python3 numpy torch

### How to run
Run the test suite:
`./spin_tests.sh`

Display help:
`python3 test.py`

Example run with PPO, hiders=1, seekers=2, epochs=3, episodes=1000, episode_length=1000, environment_type=two_walls:
`python3 test.py 1 2 ppo 3 1000 1000 two_walls`

Visualize select episodes (press or hold Enter to continue):
`python3 visualize.py data/ppo_two_walls_1-2.txt`
