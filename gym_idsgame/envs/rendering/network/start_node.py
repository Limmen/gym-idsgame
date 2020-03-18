from gym_idsgame.envs.rendering.network.node import Node

class Start(Node):

    def draw(self):
        create_circle(x * self.size + self.size / 2, y * int(self.size / 1.5) + (self.size / 1.5)/2, self.size / 7,
                      batch, background, color)