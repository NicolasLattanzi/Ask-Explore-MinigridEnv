# Ask-Explore-MinigridEnv
The minigrid environment is a perfect example of an action space with sparse reward. In particular we have most environments only giving a small reward once the goal is reached, giving no information about the performance of the agent during the training to reach it.
The agent proposed in this project is a small CNN actor-critic, which uses a PPO algorithm to simulate a trajectory to train the agent, paried with an ICM model that grants small curiosity rewards to facilitate exploration. Our goal is to present an ‘ask &amp; explore’ method that could support the curiosity exploration, giving major details about the current state of the extrinsic world.
This approach is particularly efficient with puzzle environments, such as the Door-Key env. The questions used in this project are:
- (1) Did you reach the goal?
- (2) Are you going through a hall?
- (3) Did you take a key?
- (4) Are you near a door with a key?
- (5) Did you unlock a door?
- (6) Did you fail? (ex. lava)
- (7) Have you run out of time?
Each question is assigned with a reward value, as shown in the file 'questions.py'.
Results of the project can be examined in results.txt

# Running the project
If you wish to run the project, install the requirements with
````
pip install -r requirements.txt
````
Once installed, to run the file you need to open a terminal and use one of the following:
````
python main.py --train
````
````
python main.py -e
````
Which are used respectively to train and evaluate a model in a certain environment. Be sure to check the constant MINIGRID_ENV inside of the main.py file and assign it the name of the environment you want to test.
Alternatively, the file testQA.py can be executed without any commands to manually move the agent and check the questions rewards every step.
