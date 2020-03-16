"""
Some OpenGL utility functions for drawing various shapes using the OpenGL primitive API
"""

import pyglet.gl as gl
import pyglet

def batch_label(text, x, y, font_size, color, batch, group, font_name='Times New Roman'):
    """
    Creates a text-label that can be rendered in OpenGL batch mode

    :param text: the text of the label
    :param x: the x coordinate
    :param y: the y coordinate
    :param font_size: the font size
    :param color: the color of the label
    :param batch: the batch to render the label in
    :param group: the batch group (e.g. foreground or background)
    :param font_name: the font type
    :return: a reference to the label object (in case the label has to be updated later on)
    """
    label = pyglet.text.Label(text,
                          font_name=font_name,
                          font_size=font_size,
                          x=x, y=y,
                          anchor_x='center', anchor_y='center',
                          color=color,
                          batch=batch,
                          group=group)
    return label

def batch_rect_fill(x, y, width, height, color, batch, group):
    """
    Method for rendering a filled rectangle in batch mode

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :param color: RGB color to fill the rectangle with [R,G,B] scaled between [0,1]
    :param batch: the batch to render the rectangle with
    :param group: the batch group (e.g. foreground or background)
    :return: None
    """
    color_list = list(color) + list(color) + list(color) + list(color)
    # Renders a "quad" (i.e. a shape with four sides, such as a rectangle).
    # 4 is the number of vertices (the four corners of the Quad)
    # "v2i" is the vertex format, which is 2 integers
    # the tuple list specifies four vertices (8 numbers)
    # "c3B" is the format of the color, which means RGB format 0-255
    # the color list is a list of 4*3 with a RGB color for each side of the quad
    batch.add(4, pyglet.gl.GL_QUADS, group, ('v2i', (x, y, x+width, y, x+width, y+height, x, y+height)),
              ('c3B', tuple(color_list)))

def batch_rect_border(x, y, width, height, color, batch, group):
    """
    Method for rendering a the border of a rectangle in batch mode

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :param color: RGB color to fill the rectangle with [R,G,B] scaled between [0,1]
    :param batch: the batch to render the rectangle with
    :param group: the batch group (e.g. foreground or background)
    :return: None
    """


    color_list = list(color) + list(color) + list(color) + list(color)
    # Renders the lines of a rectangle
    # 4 is the number of vertices (the four corners of the rectangle)
    # "v2i" is the vertex format, which is 2 integers
    # the tuple list specifies four vertices (8 numbers)
    # "c3B" is the format of the color, which means RGB format 0-255
    # the color list is a list of 4*3 with a RGB color for each side of the quad

    # Draw vertical lines (x,y)-->(x,y+height) and (x+width, y)-->(x+width, y+height)
    batch.add(4, pyglet.gl.GL_LINES, group,
        ('v2i', (x, y, x, y+height, x+width, y+height, x+width, y)),
        ('c3B', tuple(color_list)))

    # Draw Horizontal lines (x,y)-->(x+width,y) and (x, y+height)-->(x+width, y+height)
    batch.add(4, pyglet.gl.GL_LINES, group,
              ('v2i', (x, y, x+width, y, x, y + height, x + width, y+height)),
              ('c3B', tuple(color_list)))

def draw_and_fill_rect(x, y, width, height, color):
    """
    Draws and fills a rectangle

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :param color: RGB color to fill the rectangle with [R,G,B] scaled between [0,1]
    :return: None
    """
    __rect(x, y, width, height, color, fill=True)

def draw_rect_border(x, y, width, height, color):
    """
    Draws a rectangle with a border

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :param color: RGB color of the border [R,G,B] scaled between [0,1]
    :return: None
    """
    __rect(x,y,width,height,color)

def __rect(x, y, width, height, color, fill=False):
    """
    Draws a rectangle

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :param color: RGB color of the rectangle [R,G,B] scaled between [0,1]
    :param fill: whether to fill the rectangle or just stroke it
    :return: None
    """
    # Set the color in the OpenGL Context (State)
    gl.glColor3f(color[0], color[1], color[2])
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    # Configure rectangle (fill or not)
    if fill:
        # Delimits the vertices of a primitive or group of primitives
        gl.glBegin(gl.GL_POLYGON)

    else:
        # Delimits the vertices of a primitive or group of primitives
        gl.glBegin(gl.GL_LINES)

    # Draw the vertices of the rectangle
    __rect_vertices(x, y, width, height)
    # Delimits the vertices of a primitive or group of primitives
    gl.glEnd()

def __rect_vertices(x, y, width, height):
    """
    Uses the OpenGL API to create vertices to form a rectangle of a primitive

    :param x: the x coordinate of the lower-left  corner of the rectangle
    :param y: the y coordinate of the lower-left  corner of the rectangle
    :param width: the width of the rectangle
    :param height: the height of the rectangle
    :return: None
    """
    gl.glVertex2f(x, y)  # coordinate A
    gl.glVertex2f(x, y + height)  # coordinate B and line AB
    gl.glVertex2f(x, y + height)  # coordinate B
    gl.glVertex2f(x + width, y + height)  # coordinate C and line BC
    gl.glVertex2f(x + width, y + height)  # coordinate C
    gl.glVertex2f(x + width, y)  # coordinate D and line CD
    gl.glVertex2f(x + width, y)  # coordinate D
    gl.glVertex2f(x, y)  # coordinate A and line DA