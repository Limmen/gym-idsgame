from typing import Union

class AttackDefenseEvent:
    """
    Represents an attack-defense event to be rendered
    """
    def __init__(self, target_pos:Union[int, int], attack_defense_type:int):
        """
        Class constructor, initializes the DTO

        :param target_pos: the position of the target node of the event
        :param attack_defense_type: the type of the event
        """
        self.target_row, self.target_col = target_pos
        self.attack_defense_type = attack_defense_type