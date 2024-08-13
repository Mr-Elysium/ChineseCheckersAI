import copy
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from tqdm import tqdm, trange

from env_wrapper import WrapperEnv
from opponent import Opponent
from chinesecheckers_env import ChineseCheckersEnv
from chinesecheckers.chinesecheckers_utils import action_to_move, mirror_move

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("===== Chinese Checkers 2v2 Reinforcement Learning =====")

config_number = 1

# Load lesson for curriculum
with open(f"./configFiles/config{config_number}.yaml") as file:
   LESSON = yaml.safe_load(file)

# Define the network configuration
NET_CONFIG = {
   "arch": "cnn",  # Network architecture
   "hidden_size": [64, 64],  # Actor hidden size
   "channel_size": [128],  # CNN channel size
   "kernel_size": [4],  # CNN kernel size
   "stride_size": [1],  # CNN stride size
   "normalize": False,  # Normalize image from range [0,255] to [0,1]
}

# Define the initial hyperparameters
INIT_HP = {
   "POPULATION_SIZE": 6,
   # "ALGO": "Rainbow DQN",  # Algorithm
   "ALGO": "DQN",  # Algorithm
   "DOUBLE": True,
   # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
   "BATCH_SIZE": 256,  # Batch size
   "LR": 1e-4,  # Learning rate
   "GAMMA": 0.99,  # Discount factor
   "MEMORY_SIZE": 100000,  # Max memory buffer size
   "LEARN_STEP": 1,  # Learning frequency
   "N_STEP": 1,  # Step number to calculate td error
   "PER": False,  # Use prioritized experience replay buffer
   "ALPHA": 0.6,  # Prioritized replay buffer parameter
   "TAU": 0.01,  # For soft update of target parameters
   "BETA": 0.4,  # Importance sampling coefficient
   "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
   "NUM_ATOMS": 51,  # Unit number of support
   "V_MIN": 0.0,  # Minimum value of support
   "V_MAX": 200.0,  # Maximum value of support
   "WANDB": False,  # Use Weights and Biases tracking
}

# Define the connect four environment
env = ChineseCheckersEnv()
env.reset()

# Configure the algo input arguments
state_dim = [env.observation_space(agent)["observation"].shape for agent in env.agents]
one_hot = False
action_dim = [env.action_space(agent).n for agent in env.agents]
INIT_HP["DISCRETE_ACTIONS"] = True
INIT_HP["MAX_ACTION"] = None
INIT_HP["MIN_ACTION"] = None

# Warp the environment in the curriculum learning wrapper
env = WrapperEnv(env, LESSON)

# Pre-process dimensions for PyTorch layers
# We only need to worry about the state dim of a single agent
state_dim = np.moveaxis(np.zeros(state_dim[0]), [-1], [-3]).shape
action_dim = action_dim[0]

# Create a population ready for evolutionary hyperparameter optimisation
pop = create_population(
   INIT_HP["ALGO"],
   state_dim, #Error to be fixed in library itself
   action_dim,
   one_hot,
   NET_CONFIG,
   INIT_HP,
   population_size=INIT_HP["POPULATION_SIZE"],
   device=device,
)

# Configure the replay buffer
field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(
   memory_size=INIT_HP["MEMORY_SIZE"],  # Max replay buffer size
   field_names=field_names,  # Field names to store in memory
   device=device,
)

# Instantiate a tournament selection object (used for HPO)
tournament = TournamentSelection(
   tournament_size=2,  # Tournament selection size
   elitism=True,  # Elitism in tournament selection
   population_size=INIT_HP["POPULATION_SIZE"],  # Population size
   eval_loop=1,
)  # Evaluate using last N fitness scores

# Instantiate a mutations object (used for HPO)
mutations = Mutations(
   algo=INIT_HP["ALGO"],
   no_mutation=0.2,  # Probability of no mutation
   architecture=0,  # Probability of architecture mutation
   new_layer_prob=0.2,  # Probability of new layer mutation
   parameters=0.2,  # Probability of parameter mutation
   activation=0,  # Probability of activation function mutation
   rl_hp=0.2,  # Probability of RL hyperparameter mutation
   rl_hp_selection=[
         "lr",
         "learn_step",
         "batch_size",
   ],  # RL hyperparams selected for mutation
   mutation_sd=0.1,  # Mutation strength
   # Define search space for each hyperparameter
   min_lr=0.0001,
   max_lr=0.01,
   min_learn_step=1,
   max_learn_step=120,
   min_batch_size=8,
   max_batch_size=64,
   arch=NET_CONFIG["arch"],  # MLP or CNN
   rand_seed=1,
   device=device,
)

# Define training loop parameters
episodes_per_epoch = 10
max_episodes = LESSON["max_train_episodes"]  # Total episodes
max_steps = 500  # Maximum steps to take in each episode
evo_epochs = 20  # Evolution frequency
evo_loop = 50  # Number of evaluation episodes
elite = pop[0]  # Assign a placeholder "elite" agent
epsilon = 1.0  # Starting epsilon value
eps_end = 0.1  # Final epsilon value
eps_decay = 0.9998  # Epsilon decays
opp_update_counter = 0
wb = INIT_HP["WANDB"]

if LESSON["pretrained_path"] is not None:
    for agent in pop:
        # Load pretrained checkpoint
        agent.loadCheckpoint(LESSON["pretrained_path"])
        # Reinit optimizer for new task
        agent.lr = INIT_HP["LR"]
        agent.optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.lr)

if LESSON["opponent"] == "self":
   # Create initial pool of opponents
   opponent_pool = deque(maxlen=LESSON["opponent_pool_size"])
   for _ in range(LESSON["opponent_pool_size"]):
      opp = copy.deepcopy(pop[0])
      opp.actor.load_state_dict(pop[0].actor.state_dict())
      opp.actor.eval()
      opponent_pool.append(opp)

if max_episodes > 0:
   if wb:
      wandb.init(
         # set the wandb project where this run will be logged
         project="AgileRL",
         name="{}-EvoHPO-{}-{}Opposition-CNN-{}".format(
            "chinese_checkers_2v2",
            INIT_HP["ALGO"],
            LESSON["opponent"],
            datetime.now().strftime("%m%d%Y%H%M%S"),
         ),
         # track hyperparameters and run metadata
         config={
            "algo": "Evo HPO Rainbow DQN",
            "env": "chinese_checkers_2v2",
            "INIT_HP": INIT_HP,
            "lesson": LESSON,
         },
      )

total_steps = 0
total_episodes = 0
pbar = trange(int(max_episodes / episodes_per_epoch))

# Training loop
for idx_epi in pbar:
    turns_per_episode = []
    train_actions_hist = [0] * action_dim
    for agent in pop:  # Loop through population
        for episode in range(episodes_per_epoch):
            env.reset()  # Reset environment at start of episode
            observation, env_reward, done, truncation, _ = env.last()

            (
                p1_state,
                p1_state_flipped,
                p1_action,
                p1_next_state,
                p1_next_state_flipped,
            ) = (None, None, None, None, None)

            if LESSON["opponent"] == "self":
                # Randomly choose opponent from opponent pool if using self-play
                opponent = random.choice(opponent_pool)
            else:
                # Create opponent of desired difficulty
                opponent = Opponent(env, difficulty=LESSON["opponent"])

            # Randomly decide whether agent will go first or second
            if random.random() > 0.5:
                opponent_first = False
            else:
                opponent_first = True

            score = 0
            turns = 0  # Number of turns counter

            for idx_step in range(max_steps):
                # Player 0"s turn
                p0_action_mask = observation["action_mask"]
                p0_state = observation["observation"]
                p0_state_flipped = np.expand_dims(np.flip(p0_state, [1, 2]), 0)
                p0_state = np.expand_dims(p0_state, 0)

                if opponent_first:
                    if LESSON["opponent"] == "self":
                        p0_action = opponent.get_action(p0_state, 0, p0_action_mask)[0]
                    elif LESSON["opponent"] == "random":
                        p0_action = opponent.get_action(p0_action_mask)
                else:
                    p0_action = agent.get_action(p0_state, epsilon, p0_action_mask)[0]  # Get next action from agent
                    train_actions_hist[p0_action] += 1

                env.step(p0_action)  # Act in environment
                observation, env_reward, done, truncation, _ = env.last()
                p0_next_state = observation["observation"]
                p0_next_state_flipped = np.expand_dims(np.flip(p0_next_state, [1, 2]), 0)
                p0_next_state = np.expand_dims(p0_next_state, 0)

                if not opponent_first:
                    score += env_reward
                turns += 1

                # Check if game is over (Player 0 win)
                if done or truncation:
                    reward = env.reward(done=True, player=0, move=action_to_move(p0_action))
                    memory.save_to_memory_vect_envs(
                        np.concatenate((p0_state, p1_state, p0_state_flipped, p1_state_flipped)),
                        [action_to_move(p0_action), action_to_move(p1_action), mirror_move(action_to_move(p0_action)), mirror_move(action_to_move(p1_action))],
                        [
                            reward,
                            LESSON["rewards"]["lose"],
                            reward,
                            LESSON["rewards"]["lose"],
                        ],
                        np.concatenate(
                            (
                                p0_next_state,
                                p1_next_state,
                                p0_next_state_flipped,
                                p1_next_state_flipped,
                            )
                        ),
                        [done, done, done, done],
                    )
                else:  # Play continues
                    if p1_state is not None:
                        reward = env.reward(done=False, player=1, move=action_to_move(p1_action))
                        memory.save_to_memory_vect_envs(
                            np.concatenate((p1_state, p1_state_flipped)),
                            [action_to_move(p1_action), mirror_move(action_to_move(p1_action))],
                            [reward, reward],
                            np.concatenate((p1_next_state, p1_next_state_flipped)),
                            [done, done],
                        )

                    # Player 1"s turn
                    p1_action_mask = observation["action_mask"]
                    # Swap pieces so that the agent always sees the board from the same perspective
                    p1_state = np.flip(observation["observation"], [1, 2])
                    p1_state_flipped = np.expand_dims(np.flip(p1_state, 2), 0)
                    p1_state = np.expand_dims(p1_state, 0)

                    if not opponent_first:
                        if LESSON["opponent"] == "self":
                            p1_action = opponent.get_action(p1_state, 0, p1_action_mask)[0]
                        elif LESSON["opponent"] == "random":
                            p1_action = opponent.get_action(p1_action_mask)
                    else:
                        p1_action = agent.get_action(p1_state.copy(), epsilon, p1_action_mask)[0]  # Get next action from agent
                        train_actions_hist[p1_action] += 1

                    env.step(p1_action)  # Act in environment
                    observation, env_reward, done, truncation, _ = env.last()
                    # Swap pieces so that the agent always sees the board from the same perspective
                    p1_next_state = np.flip(observation["observation"], [1, 2])
                    p1_next_state_flipped = np.expand_dims(np.flip(p1_next_state, 2), 0)
                    p1_next_state = np.expand_dims(p1_next_state, 0)

                    if opponent_first:
                        score += env_reward
                    turns += 1

                    # Check if game is over (Player 1 win)
                    if done or truncation:
                        reward = env.reward(done=True, player=1, move=action_to_move(p1_action))
                        memory.save_to_memory_vect_envs(
                            np.concatenate((p0_state, p1_state, p0_state_flipped, p1_state_flipped)),
                            [action_to_move(p0_action), action_to_move(p1_action), mirror_move(action_to_move(p0_action)), mirror_move(action_to_move(p1_action))],
                            [
                                LESSON["rewards"]["lose"],
                                reward,
                                LESSON["rewards"]["lose"],
                                reward,
                            ],
                            np.concatenate(
                                (
                                    p0_next_state,
                                    p1_next_state,
                                    p0_next_state_flipped,
                                    p1_next_state_flipped,
                                )
                            ),
                            [done, done, done, done],
                        )

                    else:  # Play continues
                        reward = env.reward(done=False, player=0, move=action_to_move(p0_action))
                        memory.save_to_memory_vect_envs(
                            np.concatenate((p0_state, p0_state_flipped)),
                            [action_to_move(p0_action), mirror_move(action_to_move(p0_action))],
                            [reward],
                            np.concatenate((p0_next_state, p0_next_state_flipped)),
                            [done, done],
                        )

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (len(memory) >= agent.batch_size):
                    pass
                    # Sample replay buffer
                    # Learn according to agent's RL algorithm
                    #experiences = memory.sample(agent.batch_size)
                    #agent.learn(experiences)
                # TODO: uncomment. Problem with underlying library

                # Stop episode if any agents have terminated
                if done or truncation:
                    break

            total_steps += idx_step + 1
            total_episodes += 1
            turns_per_episode.append(turns)
            # Save the total episode reward
            agent.scores.append(score)

            if LESSON["opponent"] == "self":
                if (total_episodes % LESSON["opponent_upgrade"] == 0) and ((idx_epi + 1) > evo_epochs):
                    elite_opp, _, _ = tournament._elitism(pop)
                    elite_opp.actor.eval()
                    opponent_pool.append(elite_opp)
                    opp_update_counter += 1

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

    mean_turns = np.mean(turns_per_episode)

    # Now evolve population if necessary
    if (idx_epi + 1) % evo_epochs == 0:
        # Evaluate population vs random actions
        fitnesses = []
        win_rates = []
        eval_actions_hist = [0] * action_dim  # Eval actions histogram
        eval_turns = 0  # Eval turns counter
        for agent in pop:
            with torch.no_grad():
                rewards = []
                for i in range(evo_loop):
                    env.reset()  # Reset environment at start of episode
                    observation, reward, done, truncation, _ = env.last()

                    player = -1  # Tracker for which player's turn it is

                    # Create opponent of desired difficulty
                    opponent = Opponent(env, difficulty=LESSON["eval_opponent"])

                    # Randomly decide whether agent will go first or second
                    if random.random() > 0.5:
                        opponent_first = False
                    else:
                        opponent_first = True

                    score = 0

                    for idx_step in range(max_steps):
                        action_mask = observation["action_mask"]
                        if player < 0:
                            if opponent_first:
                                if LESSON["eval_opponent"] == "random":
                                    action = opponent.get_action(action_mask)
                                else:
                                    action = opponent.get_action(player=0)
                            else:
                                state = observation["observation"]
                                state = np.expand_dims(state, 0)
                                action = agent.get_action(state.copy(), 0, action_mask)[0]  # Get next action from agent
                                eval_actions_hist[action] += 1
                        if player > 0:
                            if not opponent_first:
                                if LESSON["eval_opponent"] == "random":
                                    action = opponent.get_action(action_mask)
                                else:
                                    action = opponent.get_action(player=1)
                            else:
                                # Swap pieces so that the agent always sees the board from the same perspective
                                state = np.flip(observation["observation"], [1, 2])
                                state = np.expand_dims(state, 0)
                                action = agent.get_action(state.copy(), 0, action_mask)[0]  # Get next action from agent
                                eval_actions_hist[action] += 1

                        env.step(action)  # Act in environment
                        observation, reward, done, truncation, _ = env.last()

                        if (player > 0 and opponent_first) or (player < 0 and not opponent_first):
                            score += reward

                        eval_turns += 1

                        if done or truncation:
                            break

                        player *= -1

                    rewards.append(score)
            mean_fit = np.mean(rewards)
            agent.fitness.append(mean_fit)
            fitnesses.append(mean_fit)

        eval_turns = eval_turns / len(pop) / evo_loop

        pbar.set_postfix_str(f"    Train Mean Score: {np.mean(agent.scores[-episodes_per_epoch:])}   Train Mean Turns: {mean_turns}   Eval Mean Fitness: {np.mean(fitnesses)}   Eval Best Fitness: {np.max(fitnesses)}   Eval Mean Turns: {eval_turns}   Total Steps: {total_steps}")
        pbar.update(0)

        # Format action histograms for visualisation
        train_actions_hist = [freq / sum(train_actions_hist) for freq in train_actions_hist]
        eval_actions_hist = [freq / sum(eval_actions_hist) for freq in eval_actions_hist]
        train_actions_dict = {f"train/action_{index}": action for index, action in enumerate(train_actions_hist)}
        eval_actions_dict = {f"eval/action_{index}": action for index, action in enumerate(eval_actions_hist)}

        if wb:
            wandb_dict = {
                "global_step": total_steps,
                "train/mean_score": np.mean(agent.scores[-episodes_per_epoch:]),
                "train/mean_turns_per_game": mean_turns,
                "train/epsilon": epsilon,
                "train/opponent_updates": opp_update_counter,
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
                "eval/mean_turns_per_game": eval_turns,
            }
            wandb_dict.update(train_actions_dict)
            wandb_dict.update(eval_actions_dict)
            wandb.log(wandb_dict)

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

if max_episodes > 0:
    if wb:
        wandb.finish()

# Save the trained agent
save_path = LESSON["save_path"]
os.makedirs(os.path.dirname(save_path), exist_ok=True)
elite.save_checkpoint(save_path)
print(f"Elite agent saved to '{save_path}'.")

