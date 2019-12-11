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
    None
        * AC
        python3 visualize.py WorldData/ac_none_1-2.txt 
        Seeker find hiders at corner, but Hiders avoid seeker's to another corner, mixed result

        * PPO
        python3 visualize.py WorldData/ppo_none_1-2.txt
        seekers able to keep exploring with both agents highest reward

    2Walls
        * REINFORCE
        python3 visualize.py WorldData/ac_two_walls_1-2.txt 950
        Converge to exploit corners, seekers also decide to stay at corner

        * PPO
        python3 visualize.py WorldData/ppo_two_walls_1-2.txt 250 & 750
        Seekers knows how to corner, but hiders also know how to switch behind walls



2 v 1
    None
        * AC
        python3 visualize.py WorldData/ac_none_2-1.txt
        hiders both stay at one corner, and seeker learn about corner each time

        * PPO
         python3 visualize.py WorldData/ppo_none_2-1.txt
         Seekers find different corner hider always splitting, seeker win

    2Walls
        * REINFORCE
        python3 visualize.py WorldData/reinforce_two_walls_2-1.txt 500
        Hiders know how to stay in corner, and seeker has a hard time finding them, seekers try to stay at corner
    
        * PPO
        python3 visualize.py WorldData/ppo_two_walls_2-1.txt 500
        Seekers quickly find place, each hiders want to avoid seeker, but more movement cause them to be easily spotted






