import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import minigrid
import questions

from minigrid.wrappers import ActionBonus


class Policy(nn.Module):

    def __init__(self, minigrid_env):
        super(Policy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # checking environment
        self.env_name = minigrid_env
        self.env = gym.make(minigrid_env, render_mode=None)
        # obs, _ = self.env.reset()
        # self.build_grid()
        self.env.close()
        
        self.qa = questions.QA()
        
        # === ActorCritic CNN ===
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU()
        )
        
        self.actor_logits = nn.Linear(128, 7)
        self.critic = nn.Linear(128, 1)
        
        # === CURIOSITY MODULE (ICM) ===
        self.feature_dim = 128
        self.icm = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, self.feature_dim),
            nn.ReLU()
        )
        
        # Forward model: predicts next_feature from current_feature + action
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_dim + 7, self.feature_dim),  # +7 per discrete actions one-hot
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        
        # Buffers PPO + curiosity
        self.values, self.states, self.actions = [], [], []
        self.rewards, self.log_probs, self.dones = [], [], []
        self.next_states = []
        self.intrinsic_rewards = []

    def extract_features(self, x):
        x = self.preprocessing(x)
        x = self.icm(x)
        return x

    def forward(self, x):
        x = self.preprocessing(x)
        x = self.cnn(x)
        
        logits = self.actor_logits(x)
        value = self.critic(x).squeeze(-1)
        return logits, value
    
    def build_grid(self):
        envgrid = self.env.unwrapped.grid.encode()
        w = self.env.unwrapped.width
        h = self.env.unwrapped.height
        
        self.grid = [ [0 for _ in range(w)] for _ in range(h) ]
        for row in range(h):
            for column in range(7):
                grid_value = envgrid[row, column]
                grid_value = list(map(lambda x: int(x), grid_value))
                self.grid[row][column] = [grid_value[0], grid_value[1], grid_value[2]]
    
    def compute_intrinsic_reward(self, state, action, next_state):
        state_t = self.preprocessing(state)
        next_t = self.preprocessing(next_state)
        
        with torch.no_grad():
            state_feat = self.cnn(state_t).squeeze()
            next_feat = self.cnn(next_t).squeeze()
            
            # One-hot action
            action_onehot = F.one_hot(torch.tensor(action, device=self.device), 7).float()
            
            forward_input = torch.cat([state_feat, action_onehot], dim=0)
            pred_next_feat = self.forward_model(forward_input)
            intrinsic_reward = F.mse_loss(pred_next_feat, next_feat).item()
            
        return intrinsic_reward
    
    def preprocessing(self, x):
        if type(x) == dict:
            x = x['image']
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.ndim == 3: 
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        return x.to(self.device)
    
    def act(self, state):
        with torch.no_grad():
            logits, value = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return int(action.item())
    
    
    def new_trajectory(self, rollout_steps, eta, using_qa):
        self.values, self.states, self.actions = [], [], []
        self.rewards, self.log_probs, self.dones = [], [], []
        self.next_states, self.intrinsic_rewards = [], []
        last_position = (0,0)
        still_frames = 0

        s, _ = self.env.reset()
        self.qa.build_grid(self.env)
        for t in range(rollout_steps):
            state_t = self.preprocessing(s)

            with torch.no_grad():
                logits, value = self.forward(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                a_env = int(action.item())

            new_state, reward, terminated, truncated, info = self.env.step(a_env)
            done = terminated or truncated

            # === QA REWARD ===
            if using_qa:
                reward += self.qa.check_questions(self.env, new_state, reward, done)
            
            # === CURIOSITY REWARD ===
            intr_reward = self.compute_intrinsic_reward(s, a_env, new_state)
            self.intrinsic_rewards.append(intr_reward)
            reward += eta * intr_reward  # mix extrinsic + intrinsic
            
            # === BEHAVIOR REWARD ===
            curr_position = self.env.unwrapped.agent_pos
            curr_position = list(map(lambda x: int(x), curr_position))
            
            # don't stop for more than 2 frames
            if curr_position == last_position:
                still_frames += 1
                if still_frames >= 3:
                    reward -= 0.2
            else: 
                still_frames = 0
            last_position = curr_position

            self.values.append(value.item())
            self.states.append(s)
            self.actions.append(a_env)
            self.rewards.append(reward)
            self.log_probs.append(log_prob.item())
            self.dones.append(done)
            self.next_states.append(new_state)

            s = new_state
            if done:
                s, _ = self.env.reset()
                self.qa.build_grid(self.env)
                
        with torch.no_grad():
            _, value = self.forward(self.preprocessing(s))
        return value.item()
    
    # Generalized Advantage Estimation
    def compute_advantage(self, v, gamma, lambd):
        dones = np.array(self.dones, dtype=np.bool_)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [v], dtype=np.float32)

        gae = 0.0
        returns = advantages = np.zeros_like(rewards)
        
        for t in range(len(rewards)-1, -1, -1):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lambd * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages
    
    
    def train(self):
        
        ####### TRAIN PARAMETERS    ######
        
        gamma = 0.99
        lambd = 0.95
        clipping = 0.2
        learning_rate = 0.001
        
        max_iterations = 10000
        epochs = 4
        rollout_steps = 250
        batch_size = 16
        
        # curiosity params
        eta = 0.01  # curiosity weight
        icm_lr = 0.001   # ICM learning rate
        
        using_qa = False

        ###############################
        
        self.env = gym.make(self.env_name, render_mode="rgb_array")

        avg_loss, max_score = 0.0, -float("inf")
        ppo_optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        icm_optimizer = torch.optim.Adam(list(self.icm.parameters()) + list(self.forward_model.parameters()), lr=icm_lr)
        super().train(True)

        for step in range(max_iterations):
            final_val = self.new_trajectory(rollout_steps, eta, using_qa)
            target_values, adv_values = self.compute_advantage(final_val, gamma, lambd)

            if len(self.rewards) == 0:
                continue
            current_score = float(sum(self.rewards))

            # PPO training
            obs_batch = torch.stack([self.preprocessing(obs).squeeze(0) for obs in self.states]).to(self.device)
            act_batch = torch.tensor(self.actions, dtype=torch.long, device=self.device)
            prev_logp = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
            rtg_batch = torch.tensor(target_values, dtype=torch.float32, device=self.device)
            adv_batch = torch.tensor(adv_values, dtype=torch.float32, device=self.device)

            adv_mean = adv_batch.mean()
            adv_std = adv_batch.std() + 1e-8
            adv_batch = (adv_batch - adv_mean) / adv_std
            adv_batch = torch.clamp(adv_batch, -10.0, 10.0)

            data_size = obs_batch.size(0)
            perm_idx = torch.randperm(data_size, device=self.device)
            loss_history = []
            
            # ICM Training
            if len(self.next_states) >= batch_size:  # batch minimo per ICM
                icm_batch_states = torch.stack([ self.preprocessing(s).squeeze(0) for s in self.states[:-1] ])
                icm_batch_next = torch.stack([  self.preprocessing(ns).squeeze(0) for ns in self.next_states[1:] ])
                icm_batch_actions = torch.tensor(self.actions[:-1], dtype=torch.long, device=self.device)
                
                # Feature extraction
                state_feats = self.extract_features(icm_batch_states)
                next_feats = self.extract_features(icm_batch_next)
                
                # Forward model training
                actions_onehot = F.one_hot(icm_batch_actions, 7).float()
                forward_inputs = torch.cat([state_feats, actions_onehot], dim=-1)
                pred_next_feats = self.forward_model(forward_inputs)
                
                # ICM loss = feature reconstruction + forward prediction
                icm_loss = F.mse_loss(pred_next_feats, next_feats)
                
                icm_optimizer.zero_grad()
                icm_loss.backward()
                icm_optimizer.step()

            # PPO multi-epoch updates
            for _ in range(epochs):
                batch_start = 0
                while batch_start < data_size:
                    batch_end = min(batch_start + batch_size, data_size)
                    current_batch = perm_idx[batch_start:batch_end]

                    obs_slice = obs_batch[current_batch]
                    act_slice = act_batch[current_batch]
                    old_probs_slice = prev_logp[current_batch]
                    rtg_slice = rtg_batch[current_batch]
                    adv_slice = adv_batch[current_batch]

                    pol_out, val_out = self.forward(obs_slice)
                    pol_distr = torch.distributions.Categorical(logits=pol_out)
                    curr_logp = pol_distr.log_prob(act_slice)

                    entr_term = pol_distr.entropy().mean()
                    prob_mult = (curr_logp - old_probs_slice).exp()
                    
                    obj1 = prob_mult * adv_slice
                    obj2 = torch.clamp(prob_mult, 1.0 - clipping, 1.0 + clipping) * adv_slice
                    pol_obj = -torch.min(obj1, obj2).mean()
                    
                    val_obj = F.mse_loss(val_out, rtg_slice)
                    total_loss = pol_obj + 0.5 * val_obj - 0.02 * entr_term

                    ppo_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    ppo_optimizer.step()

                    loss_history.append(total_loss.item())
                    batch_start += batch_size
            
            if current_score > max_score:
                self.save(path='best_model.pt')
                max_score = current_score
                print(f"SAVED current model. reward = {current_score:.2f}")

            avg_loss = float(np.mean(loss_history)) if loss_history else avg_loss

            if step % 100 == 0:
                avg_intr_reward = np.mean(self.intrinsic_rewards) if self.intrinsic_rewards else 0
                print(f"progress: {step}/{max_iterations} "
                      f"reward: {current_score:6.2f} "
                      f"intr_r: {avg_intr_reward:6.4f} "
                      f"loss: {avg_loss:6.2f}")
            
            if step % 250 == 0:
                self.save(path=f'model_{step}.pt')

        self.env.close()



    ############################ # #       SAVE and LOAD functions          # # #########
    
    def save(self, path='model.pt'):
        torch.save(self.state_dict(), path)

    def load(self, path='best_model.pt'):
        self.load_state_dict(torch.load(path, map_location=self.device))
    
    def load_based_on_env(self, env_name):
        self.load_state_dict(torch.load('models/' + env_name + '.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
