from minigrid.manual_control import ManualControl
from pyglet.window import key
import pyglet
import gymnasium as gym
import questions
import minigrid

""" 

file used to check QA correct function with minigrid manual movement

"""


class GridController(pyglet.window.Window):
    def __init__(self, env, qa):
        super().__init__(width=800, height=600, caption="Minigrid Manual Control")
        self.env = env
        self.qa = qa
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
    def on_draw(self):
        self.clear()
        self.env.render()
        
    def on_key_press(self, symbol, modifiers):
        action = None
        
        if symbol == key.LEFT:    action = 0  # left
        elif symbol == key.RIGHT: action = 1  # right  
        elif symbol == key.UP:    action = 2  # forward
        elif symbol == key.SPACE: action = 3  # pickup
        elif symbol == key.DOWN:  action = 4  # drop
        elif symbol == key.T:     action = 5  # toggle
        elif symbol == key.ENTER: action = 6  # done
        
        if action is not None:
            print(f"Action: {action}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            bonus = self.qa.check_questions(self.env, obs, reward, terminated or truncated)
            print(f"Step reward: {reward}, QA bonus: {bonus}")
            
            if terminated or truncated:
                print("Episode done! Reset...")
                obs, _ = self.env.reset()
                self.qa.build_grid(self.env)
                print("Reset grid:")
                self.qa.print_grid()
        
        if symbol == key.ESCAPE:
            self.close()
            
    def update(self, dt):
        pass 

'''
tested envs:

"MiniGrid-Empty-16x16-v0"
"MiniGrid-LavaGapS7-v0"
"MiniGrid-FourRooms-v0"
"MiniGrid-Dynamic-Obstacles-5x5-v0"
"MiniGrid-Dynamic-Obstacles-Random-6x6-v0"
"MiniGrid-DoorKey-6x6-v0"
"MiniGrid-DoorKey-8x8-v0"

"MiniGrid-KeyCorridorS3R2-v0"
'''

MINIGRID_ENV = "MiniGrid-LavaGapS7-v0"

def main():
    env = gym.make(MINIGRID_ENV, render_mode="human") 
    obs, _ = env.reset()
    
    QA_program = questions.QA()
    QA_program.build_grid(env)
    QA_program.print_grid()
    
    controller = GridController(env, QA_program)
    pyglet.clock.schedule_interval(controller.update, 1/60.0)
    pyglet.app.run()
    
    env.close()

    
if __name__ == "__main__":
    main()