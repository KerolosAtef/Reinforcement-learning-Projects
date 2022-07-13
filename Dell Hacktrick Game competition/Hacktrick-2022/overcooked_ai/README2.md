## Installation

- `cd overcooked_ai/`
- `pip install -e .`

## Code Structure Overview

`overcooked_ai_py` contains:

`mdp/`:
- `overcooked_mdp.py`: main Overcooked game logic
- `overcooked_env.py`: environment classes built on top of the Overcooked mdp
- `layout_generator.py`: functions to generate random layouts programmatically

`agents/`:
- `agent.py`: location of agent classes
- `benchmarking.py`: sample trajectories of agents (both trained and planners) and load various models

`planning/`:
- `planners.py`: near-optimal agent planning logic
- `search.py`: A* search and shortest path logic

## Agents
In `hacktrick_agent.py` you will find two base classes `MainAgent()` and `OptionalAgent()`. Implement according to the following cases. 
- In single mode, implement only one of them but make sure you are testing and submitting the correct agent.
- In collaborative mode, implement both classes if you want to implement different agent logic.
- In collaborative mode, implement one class only if you want to apply the same logic on both agents.

## Run and test your agents locally
Follow the steps in this notebook `\hacktrick_client\hackathon_tutorial.ipynb`

## Submit your solution
- In `hacktrick_agent.py` you will find two base classes `MainAgent()` and `OptionalAgent()`. Implement your logic in these classes.
- Run this command `python3 client.py --team_name=TEAM_NAME --password=PASSWORD --mode=MODE --layout=LAYOUT_NAME`.


