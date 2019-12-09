import numpy as np


def patchShadow(x_coord, y_coord, vis, world):

        def isShadow(x, y):
            return (world[x][y] == 4 or world[x][y] == 1)

        rightBound = min(x_coord+vis+1, world.shape[0])
        leftBound = max(x_coord-vis-1, -1)
        upperBound = min(y_coord+vis+1, world.shape[1])
        lowerBound = max(y_coord-vis-1, -1)

        # invert cross indices to not mess up quadrant calculations
        for x in range(leftBound, rightBound):
            if world[x][y_coord] == 1:
                world[x][y_coord] = 420
        for y in range(lowerBound, upperBound):
            if world[x_coord][y] == 1:
                world[x_coord][y] = 420

        for x in range(x_coord+1, rightBound): # Upper right
            for y in range(y_coord+1, upperBound):
                if isShadow(x-1,y) or isShadow(x,y-1) or isShadow(x-1,y-1):
                    world[x][y] = 4
        for x in range(x_coord-1, leftBound, -1): # Lower left
            for y in range(y_coord-1, lowerBound,-1):
                if isShadow(x+1,y) or isShadow(x,y+1) or isShadow(x+1,y+1):
                    world[x][y] = 4
        for x in range(x_coord+1, rightBound): # Lower right
            for y in range(y_coord-1, lowerBound,-1):
                if isShadow(x-1,y) or isShadow(x,y+1) or isShadow(x-1,y+1):
                    world[x][y] = 4
        for x in range(x_coord-1, leftBound, -1): # Upper left
            for y in range(y_coord+1, upperBound):
                if isShadow(x+1,y) or isShadow(x,y-1) or isShadow(x+1,y-1):
                    world[x][y] = 4
        # re-invert cross indices to do cross vision
        for x in range(leftBound, rightBound):
            if world[x][y_coord] == 420:
                world[x][y_coord] = 1
        for y in range(lowerBound, upperBound):
            if world[x_coord][y] == 420:
                world[x_coord][y] = 1
        for x in range(x_coord - 1, leftBound, -1): # Horizontal cross left
            if isShadow(x + 1, y_coord):
                world[x][y_coord] = 4
        for x in range(x_coord + 1, rightBound): # Horizontal cross right
            if isShadow(x - 1, y_coord):
                world[x][y_coord] = 4
        for y in range(y_coord - 1, lowerBound, -1): # Vertical cross down
            if isShadow(x_coord, y + 1):
                world[x_coord][y] = 4
        for y in range(y_coord + 1, upperBound): # Vertical cross up
            if isShadow(x_coord, y - 1):
                world[x_coord][y] = 4


world = np.zeros((10,10))
world[5][5] = 5
world[7][7] = 1
world[3][3] = 1
world[7][3] = 1
world[3][7] = 1
# cross walls
world[5][7] = 1
world[5][3] = 1
world[3][5] = 1
world[7][5] = 1

patchShadow(5,5,10,world)
print(world)
