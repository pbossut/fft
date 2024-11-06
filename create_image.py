# Author: Philippe Bossut
# Creation date: 2017/19/01
# Create, display and save generated grayscale images.
# The images are float in the [0.0, 1.0] range. This displays correctly with cv2.
# The creation of images is controled by a simple keyboard UI in the main.
#  * Numbers from '1' to '7' switch between the different images creation algorithm
#  * The parameters of the image creation are controlled by 2 points, a top and a bottom one, which positions
#    are interpreted to set the position of gradient points or the interval of the checkerboard pattern. The
#    control points position are controlled by 'wasd' for the top one and 'WASD' for the bottom one.
#  * 'i' saves the image to png using a simple counter to differentiate them. The normalized values need
#    to be multiplied by 255 to fill the 1 byte range.
#  * 'r' resets the position to top left and bottom right of the control points and the number of tiles to 1
#  * 'f'/'F' reduces/augments the number of tiles in the control range by 1
#  * 'c' flips a flag that centers the pattern or not
#  * 't' flips a flag that controls the tiling of the pattern or not
import numpy as np
import cv2
import math

def computeCenterTranslation(image, dx, dy):
  """Returns the translation required to center a (dx,dy) pattern on the image"""
  center_x = image.shape[0]//2
  center_y = image.shape[1]//2
  t_x = center_x - (center_x // dx)*dx - dx//2
  t_y = center_y - (center_y // dy)*dy - dy//2
  return t_x, t_y

def rotate(x, y, c_x, c_y, angle):
  """Rotates the given (x,y) by angle around center (c_x,c_y)"""
  dx = x - c_x
  dy = y - c_y
  r_x = dx*math.cos(angle) - dy*math.sin(angle)
  r_y = dx*math.sin(angle) + dy*math.cos(angle)
  return int(r_x + c_x), int(r_y + c_y)

def gradient(image, m0, m1):
  """Simple gradient from 0.0 at m0 to 1.0 at m1"""
  width = image.shape[0]
  height = image.shape[1]
  # Compute gradient parameters
  dx = float(m1[0] - m0[0]);
  dy = float(m1[1] - m0[1]);
  d = math.sqrt(dx*dx + dy*dy);
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      # We compute the projection of (m0,m) onto (m0,m1)
      dist = (float(col - m0[0])*dx + float(row - m0[1])*dy)/d;
      # Gradient between 0 and d, black before, white beyond
      if 0.0 <= dist <= d:
        image[row,col] = dist/d
      elif dist < 0.0:
        image[row,col] = 0.0
      else:
        image[row,col] = 1.0

def checkerboard(image, dx, dy, center_pattern, tiled, angle=0, offset_x=0, offset_y=0):
  """Draw a checkerboard with (dx, dy) tiles"""
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
    center_x = width//2
    center_y = height//2
  else:
    t_x = t_y = 0
    center_x = center_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      if tiled or ((abs(2*col-width)<=dx) and (abs(2*row-height)<=dy)):
        # Rotation
        x, y = rotate(col, row, center_x, center_y, angle)
        # Offset
        x += offset_x
        y += offset_y
        # Compute value
        parity = ((x-t_x) // dx) + ((y-t_y) // dy)
        image[row,col] = 0.0 if parity % 2 else 1.0
      else:
        image[row,col] = 0.0

def cones(image, dx, dy, center_pattern, tiled):
  """Draw cones on a (dx, dy) tiled checkerboard"""
  # Compute cones parameters
  cone_width = float(dx if dx < dy else dy)
  cone_width //= 2.0
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
  else:
    t_x = t_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      if tiled or ((abs(2*col-width)<=dx) and (abs(2*row-height)<=dy)):
        # Normalized distance to the center of the local cone
        center_x = col-t_x - ((col-t_x) // dx)*dx - dx//2;
        center_y = row-t_y - ((row-t_y) // dy)*dy - dy//2;
        d = math.sqrt(center_x*center_x + center_y*center_y)/cone_width;
        # Centered gradient to the center, 0 outside
        image[row,col] = 1.0 - d if d < 1.0 else 0.0
      else:
        image[row,col] = 0.0

def helices(image, dx, dy, center_pattern, tiled):
  """Draw helices on a (dx, dy) tiled checkerboard"""
  # Compute helices parameters
  pi_2 = 2.0 * math.pi;
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
  else:
    t_x = t_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      if tiled or ((abs(2*col-width)<=dx) and (abs(2*row-height)<=dy)):
        # Distance to the center of the local helix
        center_x = col-t_x - ((col-t_x) // dx)*dx - dx//2;
        center_y = row-t_y - ((row-t_y) // dy)*dy - dy//2;
        # Angle Ox,OM
        angle = math.atan2(center_y,center_x);
        # Gradient from 0 to 2*Pi
        image[row,col] = (angle+math.pi)/pi_2;
      else:
        image[row,col] = 0.0

def spheres(image, dx, dy, center_pattern, tiled):
  """Draw spheres on a (dx, dy) tiled checkerboard"""
  # Compute spheres parameters
  radius = float(dx if dx < dy else dy)
  radius /= 2.0
  radius_2 = radius*radius
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
  else:
    t_x = t_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      if tiled or ((abs(2*col-width)<=dx) and (abs(2*row-height)<=dy)):
        # Distance to the center of the local sphere
        center_x = col-t_x - ((col-t_x) // dx)*dx - dx//2;
        center_y = row-t_y - ((row-t_y) // dy)*dy - dy//2;
        d_2 = (center_x*center_x + center_y*center_y)/radius_2;
        # Sphere height to the center, 0 outside
        image[row,col] = math.sqrt(1.0 - d_2) if d_2 < 1.0 else 0.0
      else:
        image[row,col] = 0.0

def sines(image, dx, dy, center_pattern):
  """Draw 2D sine waves of wavelength (dx, dy)"""
  pi_2 = 2.0 * math.pi;
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
  else:
    t_x = t_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      # Compute sine in both directions
      sine_x = math.sin((col-t_x)*pi_2/dx)
      sine_y = math.sin((row-t_y)*pi_2/dy)
      # Combine to get normalized height then translate to be positive and normalize
      image[row,col] = (sine_x*sine_y + 1.0)/2.0;

def gaussian(image, dx, dy, sigma, center_pattern, tiled):
  """Draw 2D gaussian of width sigma on a (dx, dy) tiled checkerboard"""
  # Compute gaussian parameters
  radius = float(dx if dx < dy else dy)
  radius /= 2.0
  radius_2 = radius*radius
  sigma_2 = sigma*sigma
  width = image.shape[0]
  height = image.shape[1]
  # Compute center translation
  if center_pattern:
    t_x, t_y = computeCenterTranslation(image, dx, dy)
  else:
    t_x = t_y = 0
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      if tiled or ((abs(2*col-width)<=dx) and (abs(2*row-height)<=dy)):
        # Distance to the center of the local gaussian
        center_x = col-t_x - ((col-t_x) // dx)*dx - dx//2;
        center_y = row-t_y - ((row-t_y) // dy)*dy - dy//2;
        d_2 = (center_x*center_x + center_y*center_y)/radius_2;
        # Gaussian: we're skipping the scaling factor since we rescale to [0,1] anyway
        image[row,col] = math.exp(-d_2/(2.0*sigma_2))
      else:
        image[row,col] = 0.0

if __name__ == "__main__":
  do_something = True
  save_images_count = 0
  width = 512
  image = np.zeros((width, width), np.float32)
  x0 = 10
  y0 = 10
  x1 = 500
  y1 = 500
  tiles = 10
  draw_type = ord('2')
  center_pattern = True
  tiled = True
  while do_something:
    dx = max((x1-x0)//tiles, 1)
    dy = max((y1-y0)//tiles, 1)
    if draw_type == ord('1'):
      gradient(image, (x0,y0), (x1,y1))
    elif draw_type == ord('2'):
      checkerboard(image, dx, dy, center_pattern, tiled)
    elif draw_type == ord('3'):
      cones(image, dx, dy, center_pattern, tiled)
    elif draw_type == ord('4'):
      helices(image, dx, dy, center_pattern, tiled)
    elif draw_type == ord('5'):
      spheres(image, dx, dy, center_pattern, tiled)
    elif draw_type == ord('6'):
      sines(image, dx, dy, center_pattern)
    elif draw_type == ord('7'):
      gaussian(image, dx, dy, 0.25, center_pattern, tiled)
    cv2.imshow('image',image)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
      do_something = False
    elif key == ord('i'):
      image_name = 'Image_{0:d}.png'.format(save_images_count)
      print('Save image : ', image_name)
      cv2.imwrite(image_name, image*255)
      save_images_count += 1
    elif ord('1') <= key <= ord('7'):
      draw_type = key
    elif key == ord('w'):
      y0 -= 10
    elif key == ord('a'):
      x0 -= 10
    elif key == ord('s'):
      y0 += 10
    elif key == ord('d'):
      x0 += 10
    elif key == ord('W'):
      y1 -= 10
    elif key == ord('A'):
      x1 -= 10
    elif key == ord('S'):
      y1 += 10
    elif key == ord('D'):
      x1 += 10
    elif key == ord('r'):
      x0 = y0 = 0
      x1 = y1 = width
      tiles = 1
    elif key == ord('f'):
      tiles = max(1, tiles-1)
    elif key == ord('F'):
      tiles += 1
    elif key == ord('c'):
      center_pattern = not center_pattern
    elif key == ord('t'):
      tiled = not tiled

  cv2.destroyAllWindows()
