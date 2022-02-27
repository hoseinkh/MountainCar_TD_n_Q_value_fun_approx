###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
# code we already wrote
# import q_learning
###############################################################################
# from q_learning import plot_cost_to_go, FeatureTransformer, Model, plot_running_avg
# in this simulations we first generate some samples, learns an scaler on them and then ...
# ... perform RL on the new samples as they arrived. Hence the states (observations).
class FeatureTransformer:
  def __init__(self, env, n_components=500):
    # generate states (observations)
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    # define scaler and scale the states (observations) --> mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    #
    # Now we basically use RBF to for feature generation
    # Each RBFSampler takes each (original) (feature representation) of ...
    # ... a state and converts it to "n_components" new featuers.
    # Hence, after concatenating the new features, we convert each state to ...
    # ... {(# RBF samplers) * n_components} new features.
    #
    # We use RBF kernels with different variances to cover different parts ...
    # ... of the space.
    #
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    # For all the generated samples, transform original state representaions ...
    # ... to a new state representation using "featurizer"
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    #
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  ######################################
  def transform(self, observations):
    #
    scaled_original_state_representation = self.scaler.transform(observations)
    #
    scaled_higher_dimensions_state_representation = self.featurizer.transform(scaled_original_state_representation)
    return scaled_higher_dimensions_state_representation
###############################################################################
# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.SGDRegressors_approximate_Q_values_for_different_actions = []
    self.feature_transformer = feature_transformer
    for curr_action in range(env.action_space.n):
      # for each action, we fit a model to learn the corresponding Q-values for that action.
      curr_SGDRegressor_approximate_Q_values_for_curr_action = SGDRegressor(learning_rate=learning_rate)
      # we choose initial values high comparing to the typical rewards. This allows us ...
      # ... to use other exploration ideas, such as optimistic initial values, ...
      # ... , besides the epsiolon greedy algorithm!
      initial_optimistic_target_value = 0
      curr_SGDRegressor_approximate_Q_values_for_curr_action.partial_fit(feature_transformer.transform( [env.reset()] ), [initial_optimistic_target_value])
      self.SGDRegressors_approximate_Q_values_for_different_actions.append(curr_SGDRegressor_approximate_Q_values_for_curr_action)
  ######################################
  def predict(self, s):
    X_higher_dimension_representation_of_state_s = self.feature_transformer.transform([s])
    list_Q_values_for_state_s_and_different_actions = []
    for curr_action_index in range(len(self.SGDRegressors_approximate_Q_values_for_different_actions)):
      # for each action find the current estimate of the Q-value for each action ...
      # ... using the SGDRegressors, which are linear functions; however, we use ...
      # ... the extended feature representation instead of the original representation!
      curr_SGDRegressor_approximate_Q_values_for_curr_action = self.SGDRegressors_approximate_Q_values_for_different_actions[curr_action_index]
      curr_reward_for_curr_action_at_state_s = curr_SGDRegressor_approximate_Q_values_for_curr_action.predict(X_higher_dimension_representation_of_state_s)[0]
      list_Q_values_for_state_s_and_different_actions.append(curr_reward_for_curr_action_at_state_s)
    #
    return list_Q_values_for_state_s_and_different_actions
  ######################################
  def update(self, s, curr_action_index, G):
    X_higher_dimension_representation_of_state_s = self.feature_transformer.transform([s])
    # now we are going to update the SGDRegressor approximator for the Q-values for the action "curr_action_index"
    self.SGDRegressors_approximate_Q_values_for_different_actions[curr_action_index].partial_fit(X_higher_dimension_representation_of_state_s, [G])
  ######################################
  def epsilon_greedy_action_selection(self, s, eps):
    # Here we do epsilon greedy policy
    # Important: we can set eps = 0 so that it becomes purely greedy, ...
    # ... and we still achieve exploration! The reason is that ...
    # ... we use "optimistic initial values", and hence exploration ...
    # automatically happens even without doing explicit exploration ...
    # ... in the policy selection!
    #
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))
###############################################################################
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.ylabel("Running Average Total Reward")
  plt.xlabel("Episode")
  # plt.show()
  plt.savefig("./figs/Running_Average_Total_Reward.png")
  plt.close()
###############################################################################
# here we plot the negative of the optimal state value functions (i,e, -V*(s))!
# Note that the optimal action values are equal to the negative of the average optimal time ...
# ... that it takes to reach the mountain.
# Hence this plot shows the average optimal time to reach the top of the mountain at each state.
def plot_avg_num_remaining_steps(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -1*np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)
  #
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Num steps to reach mountain == -V(s)')
  ax.set_title("Num steps to Reach Mountain Function")
  fig.colorbar(surf)
  fig.savefig("./figs/Num_steps_to_Reach_Mountain.png")
  # plt.show()
  plt.close()
###############################################################################
# class SGDRegressor:
#   def __init__(self, **kwargs):
#     self.w = None
#     self.lr = 1e-2
#
#   def partial_fit(self, X, Y):
#     if self.w is None:
#       D = X.shape[1]
#       self.w = np.random.randn(D) / np.sqrt(D)
#     self.w += self.lr*(Y - X.dot(self.w)).dot(X)
#
#   def predict(self, X):
#     return X.dot(self.w)
##
# # replace SKLearn Regressor
# q_learning.SGDRegressor = SGDRegressor
###############################################################################
def play_one(model, eps, discount_rate, num_steps_for_TD_n=5):
  print("H1")
  observation = env.reset()
  print("observation = {}".format(observation))
  done = False
  totalreward = 0
  rewards = []
  states = []
  actions = []
  iters = 0
  # list of discount_rates for the next hypothetical steps: [discount_rate^0, discount_rate^1, ..., discount_rate^(n-1)]
  discount_rate_multipliers = []
  for i in range(num_steps_for_TD_n):
    discount_rate_multipliers.append(discount_rate**i)
  discount_rate_multipliers = np.array(discount_rate_multipliers)
  #
  # while the episode is not finished or we have limited number of iterations:
  while not done and iters < 10000:
    # randomly pick an actio
    action = model.epsilon_greedy_action_selection(observation, eps)
    # record the observation and the corrresponding action
    states.append(observation)
    actions.append(action)
    #
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    # we record the rewards as we move forward because we need to have ...
    # ... a list of rewards (for the next steps) in order to calculate ...
    # ... the rewards.
    rewards.append(reward)
    #
    # # update the model
    # if we have at least n rewards, then keep updating the states!
    if len(rewards) >= num_steps_for_TD_n:
      #
      # total reward: G  = sum of discounted rewards for the past num_steps_for_TD_n steps (i.e. return_up_to_prediction) ...
      # ... plus the discounted Q-value afterwards i.e. ((discount_rate**num_steps_for_TD_n)*np.max(model.predict(observation)[0]))
      return_up_to_prediction = discount_rate_multipliers.dot(rewards[-num_steps_for_TD_n:])
      G = return_up_to_prediction + (discount_rate**num_steps_for_TD_n)*np.max(model.predict(observation)[0])
      # update the state-action value functions (if tabular) or update Q-value function approximator
      model.update(states[-num_steps_for_TD_n], actions[-num_steps_for_TD_n], G)
    #
    totalreward += reward
    iters += 1
  #
  # the episode is finished; hence empty the cache
  if num_steps_for_TD_n == 1: # i.e. TD(0)
    rewards = []
    states = []
    actions = []
  else: #i.e. keep the last (num_steps_for_TD_n - 1) rewards, states, and actions.
    rewards = rewards[-num_steps_for_TD_n + 1:]
    states = states[-num_steps_for_TD_n + 1:]
    actions = actions[-num_steps_for_TD_n + 1:]
  #
  # check to see if we successfully reached the mountain
  if observation[0] >= 0.5:
    # we reached the mountain
    while len(rewards) > 0:
      # update the Q-values for ALL the (state, action) pairs ...
      # ... in this finished episode
      G = discount_rate_multipliers[:len(rewards)].dot(rewards)
      model.update(states[0], actions[0], G)
      rewards.pop(0)
      states.pop(0)
      actions.pop(0)
  else:
    # we did not reach the goal and the gym stopped the episode.
    while len(rewards) > 0:
      # update the Q-values for ALL the (state, action) pairs ...
      # ... in this terminated episode
      # Because this episode is terminated, we guess the remaining steps would have ...
      # ... been unsuccessfull, and hence would have get -1 reward
      guess_rewards = rewards + [-1 for j in range(num_steps_for_TD_n - len(rewards))]
      G = discount_rate_multipliers.dot(guess_rewards)
      model.update(states[0], actions[0], G)
      rewards.pop(0)
      states.pop(0)
      actions.pop(0)
  #
  return totalreward
###############################################################################
if __name__ == '__main__':
  env = gym.make('MountainCar-v0').env
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  discount_rate = 0.99
  num_steps_for_TD_n = 5
  #
  if True:
    monitor_dir = os.getcwd() + "/videos/" + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  num_of_episodes = 300
  totalrewards = np.empty(num_of_episodes)
  costs = np.empty(num_of_episodes)
  for i in range(num_of_episodes):
    # curr_eps = 0
    curr_eps = 0.1*(0.97**i)
    totalreward = play_one(model, curr_eps, discount_rate, num_steps_for_TD_n)
    totalrewards[i] = totalreward
    print("episode:", i, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())
  #
  plt.plot(totalrewards)
  plt.xlabel("Episode")
  plt.ylabel("Rewards")
  plt.savefig("./figs/Average_Total_Reward.png")
  plt.close()
  #
  plot_running_avg(totalrewards)
  #
  # plot the optimal state-value function
  plot_avg_num_remaining_steps(env, model)
  #

