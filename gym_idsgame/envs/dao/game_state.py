
class GameState:

    def __init__(self):
        self.game_step = 0
        self.num_games = 0
        self.attack_type = 1
        self.done = False
        self.defense_events = []
        self.attack_events = []