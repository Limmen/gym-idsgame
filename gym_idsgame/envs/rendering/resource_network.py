from gym_idsgame.envs.rendering.resource_node import ResourceNode
from gym_idsgame.envs.rendering.render_util import batch_label, batch_rect_fill, batch_line

class ResourceNetwork:
    """
    Class representing the resource network in the rendering
    """
    def __init__(self, cell_size, num_rows, num_cols):
        """
        Class constructor, initializes the resource network and the state

        :param cell_size: size of an individual cell in the grid
        :param num_rows: number of rows in the grid
        :param num_cols: number of columns in the grid
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cell_size = cell_size
        self.grid = [[ResourceNode(cell_size) for j in range(self.num_cols)] for i in range(self.num_rows)]

    def get_cell(self, row, col):
        """
        Gets a specific cell from the grid

        :param row: the row of the cell in the grid
        :param col: the column of the cell in the grid
        :return: the cell at the given row and column
        """
        return self.grid[row][col]


    def connect_start_and_server_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n1.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n2.get_link_coords(upper=True, lower=False)
        batch_line(x1, y1, x2, y1, color, batch, group, line_width)
        batch_line(x2, y1, x2, y2, color, batch, group, line_width)
        if row1 == self.num_rows-1 and col1 == col2:
            batch_line(x1, y1+self.cell_size/6, x2, y2, color, batch, group, line_width)

    def connect_server_and_server_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n2.get_link_coords(lower=True, upper=False)
        x2, y2, col2, row2 = n1.get_link_coords(upper=True, lower=False)
        batch_line(x2, y1, x2, y2, color, batch, group, line_width)

    def connect_server_and_data_nodes(self, n1, n2, color, batch, group, line_width):
        x1, y1, col1, row1 = n2.get_link_coords()
        x2, y2, col2, row2 = n1.get_link_coords(upper=False, lower=True)
        batch_line(x1, y1, x2, y1, color, batch, group, line_width)
        batch_line(x2, y1, x2, y2, color, batch, group, line_width)
        if row1 == 0 and col1 == col2:
            batch_line(x2, y2, x2, y2-self.cell_size/3, color, batch, group, line_width)