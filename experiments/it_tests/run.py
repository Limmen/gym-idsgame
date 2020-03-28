"""
Integration test, runs 10 episodes of each experiment to verify that its basic functionality is OK.
Good to run whenever changes is made to the pre-implemented agents or environment.
"""
import os
from gym_idsgame.config.runner_mode import RunnerMode
from gym_idsgame.simulation.dao.simulation_config import SimulationConfig
from gym_idsgame.agents.dao.agent_type import AgentType
from gym_idsgame.config.client_config import ClientConfig
from gym_idsgame.agents.dao.q_agent_config import QAgentConfig
from gym_idsgame.runnner import Runner


def default_output_dir() -> str:
    script_dir = os.path.dirname(__file__)
    return script_dir


def test_sim_attack_maximal_vs_defend_minimal(version):
    simulation_config = SimulationConfig(log_frequency=1, num_episodes=10)
    env_name = "idsgame-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.ATTACK_MAXIMAL_VALUE.value,
                                 defender_type=AgentType.RANDOM.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_sim_attack_maximal_vs_random(version) -> ClientConfig:
    simulation_config = SimulationConfig(log_frequency=1, num_episodes=10)
    env_name = "idsgame-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.ATTACK_MAXIMAL_VALUE.value,
                                 defender_type=AgentType.DEFEND_MINIMAL_VALUE.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_sim_random_vs_defend_minimal(version) -> ClientConfig:
    simulation_config = SimulationConfig(log_frequency=1, num_episodes=10)
    env_name = "idsgame-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.RANDOM.value,
                                 defender_type=AgentType.DEFEND_MINIMAL_VALUE.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_sim_random_vs_random(version) -> ClientConfig:
    simulation_config = SimulationConfig(log_frequency=1, num_episodes=10)
    env_name = "idsgame-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.RANDOM.value,
                                 defender_type=AgentType.RANDOM.value, mode=RunnerMode.SIMULATE.value,
                                 simulation_config=simulation_config, output_dir=default_output_dir(),
                                 title="RandomAttacker vs RandomDefender")
    Runner.run(client_config)


def test_train_maximal_attack_tabular_q_learning(version) -> ClientConfig:
    q_agent_config = QAgentConfig(num_episodes=10, eval_frequency=100, attacker=False, defender=True)
    env_name = "idsgame-maximal_attack-v" + str(version)
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_train_minimal_defense_tabular_q_learning(version) -> ClientConfig:
    q_agent_config = QAgentConfig(num_episodes=10, eval_frequency=100, attacker=True, defender=False)
    env_name = "idsgame-minimal_defense-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_train_random_attack_tabular_q_learning(version) -> ClientConfig:
    q_agent_config = QAgentConfig(num_episodes=10, eval_frequency=100, attacker=False, defender=True)
    env_name = "idsgame-random_attack-v" + str(version)
    client_config = ClientConfig(env_name=env_name, defender_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_DEFENDER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir())
    Runner.run(client_config)


def test_train_random_defense_tabular_q_learning(version) -> ClientConfig:
    q_agent_config = QAgentConfig(num_episodes=10, eval_frequency=100, attacker=True, defender=False)
    env_name = "idsgame-random_defense-v" + str(version)
    client_config = ClientConfig(env_name=env_name, attacker_type=AgentType.TABULAR_Q_AGENT.value,
                                 mode=RunnerMode.TRAIN_ATTACKER.value,
                                 q_agent_config=q_agent_config, output_dir=default_output_dir())
    Runner.run(client_config)


# Program entrypoint
if __name__ == '__main__':
    versions = list(range(4))
    for version in versions:
        test_sim_attack_maximal_vs_defend_minimal(version)
        test_sim_attack_maximal_vs_random(version)
        test_sim_random_vs_defend_minimal(version)
        test_sim_random_vs_random(version)
        test_train_maximal_attack_tabular_q_learning(version)
        test_train_minimal_defense_tabular_q_learning(version)
        test_train_random_attack_tabular_q_learning(version)
        test_train_random_defense_tabular_q_learning(version)



