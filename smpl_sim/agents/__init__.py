from .agent import Agent
from .agent_pg import AgentPG
from .agent_ppo import AgentPPO
from .agent_humanoid import AgentHumanoid

agent_dict = {
    'agent': Agent,
    'agent_pg': AgentPG,
    'agent_ppo': AgentPPO,   
    'agent_humanoid': AgentHumanoid,
}