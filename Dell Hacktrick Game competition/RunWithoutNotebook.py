# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

from hacktrick_ai_py.agents.benchmarking import AgentEvaluator, LayoutGenerator
from hacktrick_ai_py.visualization.state_visualizer import StateVisualizer
from hacktrick_ai_py.agents.agent import AgentPair, StayAgent
from hacktrick_agent import HacktrickAgent

def run_agent(mode, timesteps, layout_name, hacktrick_agent):
  hacktrick_agent.set_mode(mode)
  if mode == 'collaborative':
    agent0 = hacktrick_agent.agent0
    agent1 = hacktrick_agent.agent1
    agent = AgentPair(agent0, agent1)
  elif mode == 'single':
    agent0 = hacktrick_agent.agent0
    agent1 = StayAgent()
    agent = AgentPair(agent0, agent1)
  mdp_gen_params = {"layout_name": layout_name}
  mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_gen_params)
  env_params = {"horizon": timesteps}
  agent_eval = AgentEvaluator(env_params=env_params, mdp_fn=mdp_fn)
  trajectories = agent_eval.evaluate_agent_pair(agent, num_games=1)
  return trajectories


# Parameters to be changed

# mode = 'single'
# timesteps = 1200
# layout_name = 'leaderboard_single'

mode = 'collaborative'
timesteps = 1200
layout_name = 'leaderboard_collaborative'

agent = HacktrickAgent()
trajectories = run_agent(mode, timesteps, layout_name, agent)



# print(trajectories)

def visualize(trajectories):
  img_dir_path = StateVisualizer().display_rendered_trajectory(trajectories, trajectory_idx=0, ipython_display=True)


visualize(trajectories)