# HideSeekRL

Useful link

Soft Actor-Critic for continuous and discrete actions(With Code)
https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954

Some backgrounds of multi-agent learning
https://bair.berkeley.edu/blog/2018/12/12/rllib/

Some questions:
What is Advantage Function?


Interesting Result about finding shadow spot to hide

python3 visualize.py data/reinforce_big_walls_2-1.txt 500

python3 test.py 1 1 actor_critic 1 50 100 none

python3 compare_data.py data/reinforce_none_1-1.txt,data/actor_critic_none_1-1.txt reinforce,actor_critic img/temp_compare.png

=================== Behavior Analysis =========================

1 v 2 
*
python3 visualize.py WorldData/ac_none_1-2.txt 
Seeker find corner, but Hiders avoid seeker's corner, mixed result
*
python3 visualize.py WorldData/ac_two_walls_1-2.txt 
Hider stay behind a corner , seekers take some time to find out and learn the new policy



1 v 1 
python3 visualize.py WorldData/ac_none_1-1.txt 
Hiders sticks at corner, that's why hider win


*
python3 visualize.py WorldData/ac_two_walls_1-1.txt
Hider Finds spot to stay to hide 



python3 visualize.py WorldData/ac_two_walls_2-1.txt 
Hider stays at each according corner, hiders have a slight more avantage

PPO - NONE






Hiders stick at corner, seekers stick at corner
python3 visualize.py WorldData/reinforce_two_walls_2-1.txt

Hiders hide behind walls
python3 visualize.py WorldData/ppo_two_walls_1-1.txt 350

Seekers Cornering Hiders, Hiders hide behind walls
python3 visualize.py WorldData/ppo_two_walls_1-2.txt 200
900



