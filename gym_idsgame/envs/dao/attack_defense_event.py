from typing import Union

class AttackDefenseEvent:
    """
    Represents an attack-defense event to be rendered
    """
    def __init__(self, target_pos:Union[int, int], attack_defense_type:int):
        self.target_row, self.target_col = target_pos
        self.attack_defense_type = attack_defense_type