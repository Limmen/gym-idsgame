from gym_idsgame.envs.rendering.grid_cell import GridCell

class Grid:
    """
    Class representing the grid in the rendering
    """
    def __init__(self, cell_size, num_rows, num_cols, goal_state_x, goal_state_y):
        """
        Class cosntructor, initializes the grid and the state

        :param cell_size: size of an individual cell in the grid
        :param num_rows: number of rows in the grid
        :param num_cols: number of columns in the grid
        :param goal_state_x: the column of the goal state
        :param goal_state_y: the  row of the goal state
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = [[GridCell(cell_size) for j in range(self.num_cols)] for i in range(self.num_rows)]
        self.grid[goal_state_y][goal_state_x].goal_state = True

    def get_cell(self, row, col):
        """
        Gets a specific cell from the grid

        :param row: the row of the cell in the grid
        :param col: the column of the cell in the grid
        :return: the cell at the given row and column
        """
        return self.grid[row][col]