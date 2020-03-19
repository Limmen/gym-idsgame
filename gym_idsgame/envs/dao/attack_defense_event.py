
class AttackDefenseEvent:
    """
    Represents an attack-defense event to be rendered
    """
    def __init__(self, target_col:int, target_row:int, attack_defense_type:int):
        self.target_col = target_col
        self.target_row = target_row
        self.attack_defense_type = attack_defense_type