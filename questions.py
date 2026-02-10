"""
Questions act as additional reward to agent's actions, mitigating the difficulty
of exploring a sparse reward environment.
The idea is that every question's base response is "No" (or False), and every time
the question becomes "Yes", the agent gets a reward. Different questions have different
reward values. (ex: reaching the goal represents maximum reward. Dying is negative reward)

True-Answers rewards must be one-shot to avoid exploitation!!!!
The environment is really simple, so the QA method can be sometimes be approximated to 
an Action-Reward program

Note: this is a simplified scenario, so the QA program is meant to be able to observe the
entire environment. Still, the logic employed in this project can still be applied to an
agent's pure observation to achieve an equal result.

Questions:
1. Did you reach the goal?
2. Are you going through a hall?
3. Did you take a key?
4. Are you near a door with a key?
5. Did you unlock a door?
6. Did you fail? (ex. lava)
7. Have you run out of time?
"""

from minigrid.core.world_object import Key

# Question Answering program

class QA():
    
    def __init__(self):
        self.grid = []
        self.carrying = False
    
    def check_questions(self, env, obs, reward, done):
        bonus = 0  # total reward for the agent
        
        carried_item = env.unwrapped.carrying
        curr_position = env.unwrapped.agent_pos
        curr_position = list(map(lambda x: int(x), curr_position))
        x, y = curr_position
        
        # 7. run out of time
        if env.unwrapped.step_count == env.unwrapped.max_steps:
            return -1
        
        if done:
            # 1. Did you reach the goal?
            if reward > 0:
                return max(5, 5 * reward)
            # 6. Did you fail?
            else:
                return -10
        
        # 2. Are you going through a hall?
        # if self.grid[x][y] == 5:
        #     bonus += 0.15
        #     self.grid[x][y] = 0
        
        # 3. Did you take a key?
        # if not self.carrying and self.has_key(carried_item):
        #     bonus += 1
        #     self.carrying = True
        
        # # 4. Are you near a door with a key?
        # elif self.grid[x][y] == 6 and self.carrying:
        #     bonus += 0.12
        
        # 5. Did you unlock a door?
        # elif self.grid[x][y] == 4:
        #     bonus += 2
        #     self.grid[x][y] = 0 # no exploit!
        #     self.grid[x-1][y] = 0 
            
        return bonus

    def has_key(self, item):
        if item is None:
            return False
        return type(item) is Key
    
    
    def build_grid(self, env):
        envgrid = env.unwrapped.grid.encode()
        #print(envgrid)
        envgrid = env.unwrapped.grid.encode()
        h, w, _ = envgrid.shape
        #w = env.unwrapped.width
        #h = env.unwrapped.height
        
        self.grid = [ [0 for _ in range(w)] for _ in range(h) ]
        for row in range(h):
            for column in range(w):
                grid_value = envgrid[row, column, :]
                grid_value = list(map(lambda x: int(x), grid_value))
                v = 0
                if grid_value[0] == 1: # floor
                    continue
                elif grid_value[0] == 2: # wall
                    v = 1
                elif grid_value[0] == 8: # goal
                    v = 2
                elif grid_value[0] == 9: # lava
                    v = 3
                elif grid_value[0] == 4: # door
                    v = 4
                self.grid[row][column] = v
        
        for row in range(h):
            for col in range(w):
                # hall/ door
                if self.grid[row][col] == 0: 
                    if self.grid[row][col+1] == 1 and self.grid[row][col-1] == 1:
                        self.grid[row][col] = 5
                    elif self.grid[row+1][col] == 1 and self.grid[row-1][col] == 1:
                        self.grid[row][col] = 5
                # in front of a door
                elif self.grid[row][col] == 4:
                    self.grid[row-1][col] = 6
                
    
    def print_grid(self):
        for row in self.grid:
            print(row)
        
