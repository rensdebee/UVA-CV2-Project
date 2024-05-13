import sys
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import numpy as np
import os

# Initialize Pygame and OpenGL
pygame.init()

def initialize_openGL(display):
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.1, 0.1, 0.1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def load_obj(file):
    vertices = []
    faces = []
    with open(file) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                faces.append(face)
    return vertices, faces

def draw_model(vertices, faces):
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def calculate_camera_position(azimuth, elevation, radius=10):
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    x = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    y = radius * np.sin(elevation_rad)
    z = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    return (x, y, z)

def render_model_at_angle(display, obj_path, azimuth, elevation, output_path):
    initialize_openGL(display)
    vertices, faces = load_obj(obj_path)
    camera_pos = calculate_camera_position(azimuth, elevation)
    glLoadIdentity()
    gluLookAt(*camera_pos, 0, 0, 0, 0, 1, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_model(vertices, faces)
    pygame.display.flip()
    pygame.image.save(pygame.display.get_surface(), output_path)

def main():
    display = (800, 600)
    angles = [
        (0, 0),     # Front view
        (180, 0),   # Back view
        (90, 0),    # Left side view
        (270, 0),   # Right side view
        (0, 90),    # Top view
        (0, -90)    # Bottom view
    ]
    obj_path = '/Users/devindewilde/Documents/GitHub/UVA-CV2-Project/logs/baseline_ism/corgi.obj'  # Update with the path to your .obj file
    os.makedirs('renders', exist_ok=True)
    for idx, (azimuth, elevation) in enumerate(angles):
        output_path = f'renders/render_{idx}.png'
        render_model_at_angle(display, obj_path, azimuth, elevation, output_path)
        print(f'Rendered image saved to {output_path}')

if __name__ == "__main__":
    main()
