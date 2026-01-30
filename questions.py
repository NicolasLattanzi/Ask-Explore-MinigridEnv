"""
Questions act as additional reward to agent's actions, mitigating the difficulty
of exploring a sparse reward environment.
The idea is that every question's base response is "No" (or False), and every time
the question becomes "Yes", the agent gets a reward. Different questions have different
reward values. (ex: reaching the goal represents maximum reward. Dying is negative reward)

True-Answers rewards must be one-shot to avoid exploitation!!!!

Questions:
1. Did you reach the goal?
2. Can you see the goal?
3. Did you take a key?
4. Did you unlock a door?
5. Are you going through a door/ hall?
6. Did you die? (ex. lava)
"""

from minigrid.core.world_object import Key

# Question Answering program

class QA():
    
    def __init__(self):
        self.grid = []
        self.carrying = False
    
    def check_questions(self, env):
        bonus = 0  # total reward for the agent
        
        carried_item = env.unwrapped.carrying
        curr_position = env.unwrapped.agent_pos
        curr_position = list(map(lambda x: int(x), curr_position))
        x, y = curr_position
        
        # 1. Did you reach the goal?
        if self.grid[x][y] == 2: # reward based on steps??
            bonus += 10
        # 6. Did you die in lava?
        elif self.grid[x][y] == 3:
            bonus -= 10
        else:
            # 3. Did you take a key?
            if not self.carrying and self.has_key(carried_item):
                bonus += 1
                self.carrying = True
            
            # 4. Did you unlock a door?
            if self.grid[x][y] == 4:
                bonus += 1
                self.grid[x][y] == 0 # no exploit!
            
            # 5. Are you going through a door/ hall?
            if self.grid[x][y+1] == 1 and self.grid[x][y-1] == 1:
                bonus += 0.1
            elif self.grid[x+1][y] == 1 and self.grid[x-1][y] == 1:
                bonus += 0.1
        
        return bonus

    def has_key(self, item):
        if item is None:
            return False
        return type(item) is Key
    
    
    def build_grid(self, env):
        envgrid = env.unwrapped.grid.encode()
        #print(envgrid)
        w = env.unwrapped.width
        h = env.unwrapped.height
        
        self.grid = [ [0 for _ in range(w)] for _ in range(h) ]
        for row in range(h):
            for column in range(w):
                grid_value = envgrid[row, column]
                grid_value = list(map(lambda x: int(x), grid_value))
                v = 0
                if grid_value == [1,0,0]: # floor
                    continue
                elif grid_value == [2,5,0]: # wall
                    v = 1
                elif grid_value == [8,1,0]: # goal
                    v = 2
                elif grid_value == [9,0,0]: # lava
                    v = 3
                elif grid_value == [4,4,2]: # door
                    v = 4
                self.grid[row][column] = v
    
    def print_grid(self):
        for row in self.grid:
            print(row)
        
