"""
The idea is to give rewards to the agent every time a question goes from 0 to 1.
Depending on the environment, not every question starts at zero.
When a question goes to 1 the agent gets a huge reward, and every True question
continues to give small rewards to the agent. When a question becomes True and
every other question is already True, the Agent gets a 10 reward.

True-Answers rewards must be one-shot to avoid exploitation!!!!

Questions:
1. Did you reach the goal?
2. Can you see the door?
3. Did you take a key?
4. Did you unlock a door?
5. Are you going through a door?
6. Did you die? (ex. lava)
7. Are you near an obstacle?
"""

#grid_value_dict = {[1,0,0]: 0, [2,5,0]: 1,}
# 0:empty cell, 1:wall

# Question Answering program

class QA():
    
    def __init__(self):
        self.questions = [0 for _ in range(3)]
        self.grid = []
    
    def update_questions(self, env):
        bonus = 0  # total reward for the agent
        
        curr_position = env.unwrapped.agent_pos
        curr_position = list(map(lambda x: int(x), curr_position))
        
        # going through doors is good!
        x, y = curr_position
        if self.grid[x][y+1] == [2,5,0] and self.grid[x][y-1] == [2,5,0]:
            reward += 0.1
        elif self.grid[x+1][y] == [2,5,0] and self.grid[x-1][y] == [2,5,0]:
            reward += 0.1
        
        for q in self.questions:
            bonus += q 
        
        return bonus
    
    
    def build_grid(self, env):
        envgrid = env.unwrapped.grid.encode()
        w = env.unwrapped.width
        h = env.unwrapped.height
        
        self.grid = [ [0 for _ in range(w)] for _ in range(h) ]
        for row in range(h):
            for column in range(7):
                grid_value = envgrid[row, column]
                grid_value = list(map(lambda x: int(x), grid_value))
                self.grid[row][column] = [grid_value[0], grid_value[1], grid_value[2]]
        
