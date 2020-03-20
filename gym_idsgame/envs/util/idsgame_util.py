from gym_idsgame.envs.dao.idsgame_config import IdsGameConfig

def validate_config(idsgame_config: IdsGameConfig) -> None:
    if idsgame_config.game_config.num_layers < 1:
        raise AssertionError("The number of layers cannot be less than 1")
    if idsgame_config.game_config.num_attack_types < 1:
        raise AssertionError("The number of attack types cannot be less than 1")
    if idsgame_config.game_config.max_value < 3:
        raise AssertionError("The max attack/defense value cannot be less than 3")