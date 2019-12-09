import sys

print_lines = True
start_episode = 0
if len(sys.argv) > 2:
    start_episode = sys.argv[2]
    print_lines = False

with open(sys.argv[1], "r") as f:
    timestep = 0
    episode = ""
    for line in f:
        if "Episode" in line:
            if line.split()[1] != episode:
                episode = line.split()[1]
                timestep = 0
            if episode == start_episode:
                print_lines = True
        elif print_lines:
            print("\n".join(line.split(",")))
            input("Episode %s, Timestep\t%d" % (episode, timestep))
            timestep += 1