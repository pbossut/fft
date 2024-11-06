# Author: Philippe Bossut
# Creation date: 2017/05/02
import numpy as np
import cv2
import math
import create_image
import fft_experiments

def GetImage(draw_type, input_size, dx, dy, angle, offset_x, offset_y):
  """Use one of the create image primitive"""
  image = np.zeros((input_size, input_size), np.float32)
  if draw_type == ord('2'):
    create_image.checkerboard(image, dx, dy, True, True, angle, offset_x, offset_y)
  elif draw_type == ord('3'):
    create_image.cones(image, dx, dy, True, True)
  elif draw_type == ord('4'):
    create_image.helices(image, dx, dy, True, True)
  elif draw_type == ord('5'):
    create_image.spheres(image, dx, dy, True, True)
  elif draw_type == ord('6'):
    create_image.sines(image, dx, dy, True)
  elif draw_type == ord('7'):
    create_image.gaussian(image, dx, dy, 0.25, True, True)
  return image

if __name__ == "__main__":
  do_something = True
  increment_size = 1
  increment_rotation = 1.0*math.pi/180.0
  # Note: dx == 18 is a very colorful value
  dx = dy = 6
  draw_type = ord('2')
  angle = 0.0
  offset_x = offset_y = 0
  # Note: input_size = 160 gives decent refresh rate
  input_size = 256
  save_count = 0
  # Note: in milliseconds, wait_time == 0 means that the code stops for key stroke
  wait_time = 0
  # Define the codec and create VideoWriter object to file Test.avi
  # Note: I tried several things before finding something that works on Mac.
  #fourCC = cv2.VideoWriter_fourcc('M','J','P', 'G')
  #out = cv2.VideoWriter('Anim.avi', fourCC, 20.0, (512,512), True)
  while do_something:
    # Compute a pattern (grayscale) - Set the angle to 0 for the moment
    #print("dx = ", dx)
    print("dx:", dx, ", dy:", dy, "offset_x:", offset_x, ", offset_y:", offset_y, ", angle:", angle*180.0/math.pi)
    image = GetImage(draw_type, input_size, dx, dy, angle, offset_x, offset_y)
    # Compute its FFT
    img_fft = fft_experiments.ComputeFft(image)
    # Compute and prepare for display the filtered module and phase
    module, phase = fft_experiments.ExtractModuleAndPhase(img_fft)
    fft_combi = fft_experiments.ConvertModulePhasetoBGR(module, phase)
    fft_combi = cv2.resize(fft_combi, (1024,1024))
    #fft_combi = cv2.resize(module, (512,512))
    # Save frame for the movie
    #out.write(fft_combi)
    # Increment for the next round
    dx += increment_size
    dy += increment_size
    # Avoid the black frames for the movie
    #if dx in [8, 10, 16, 20, 40]:
    #  dx += increment_size
    #  dy += increment_size
    #input_size += increment_size
    # using the angle to compute the offset
    #angle += increment_rotation
    #offset_x = int(dx*math.cos(angle))
    #offset_y = int(dx*math.sin(angle))
    dx = max(6,dx)
    dx = min(52,dx)
    dy = max(6,dy)
    dy = min(52,dy)
    #angle = min(math.pi/2.0,angle)
    #angle = max(-math.pi/2.0,angle)
    offset_x = min(dx*2,offset_x)
    offset_x = max(-dx*2,offset_x)
    offset_y = min(dy*2,offset_y)
    offset_y = max(-dy*2,offset_y)
    input_size = max(32,input_size)
    input_size = min(256,input_size)
    increment_size = 1 if dx == 6 else increment_size
    increment_size = -1 if dx == 52 else increment_size
    # Rotate after one cycle
    #if dx == 6:
    #  angle += increment_rotation
    #increment_rotation = -0.1*math.pi/180.0 if angle == math.pi/2.0 else increment_rotation
    #increment_rotation = 0.1*math.pi/180.0 if angle == -math.pi/2.0 else increment_rotation
    # Show the images
    cv2.imshow('image',image)
    cv2.imshow('fft',fft_combi)
    # Wait for key stroke and interpert the key
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
      do_something = False
    elif ord('2') <= key <= ord('7'):
      draw_type = key
    elif key == ord('z'):
      increment_size = -1
    elif key == ord('x'):
      increment_size = 1
    elif key == ord('c'):
      #increment_rotation = -1.0*math.pi/180.0
      angle += increment_rotation
    elif key == ord('v'):
      #increment_rotation = 1.0*math.pi/180.0
      angle -= increment_rotation
    elif key == ord('r'):
      dx = dy = 10
      angle = 0
      increment_size = 0
      increment_rotation = 0
    elif key == ord('i'):
      # Save the colorized FFT
      name = 'FFT_{0:d}.png'.format(save_count)
      print('Save FFT image : ', name)
      cv2.imwrite(name,fft_combi)
      save_count += 1

  # Save the movie
  #out.release()
  # Cleanup and exit
  cv2.destroyAllWindows()

