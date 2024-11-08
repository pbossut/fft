# Author: Philippe Bossut
# Creation date: 2017/21/01
# Idea: Create synthetic images then compute and display their FFT finding a way to display both
# the module and the phase. I think the way to go is to map the phase angle to the Hue and the
# module to the Value and set Saturation to 1.0. That way, the image will look like the usual FFT
# module image but with variation of hue depending of the phase.
# Motivation: There's no useful analysis of the phase in FFT papers and I'm interested to know what kind of
# info the phase carries. Once this works, I'll try to FFT inverse after manipulating the phase only or
# the amplitude only and see how the image is different.
# Hope: I'm trying to see if one can find a signature for images that is rotation and translation invariant,
# possibly scale invariant. That will be great to identify segmented images and provide a distance metric
# between images to qualify the identification. Note that ML (TensorFlow) could render all that completely
# moot...
import numpy as np
import cv2
import create_image
import math

def GetImage(draw_type, m0, m1, dx, dy, center_pattern, tiled, file_name):
  """Use one of the create image primitive or load the file_name image"""
  image = np.zeros((512, 512), np.float32)
  if draw_type == ord('1'):
    create_image.gradient(image, m0, m1)
  elif draw_type == ord('2'):
    create_image.checkerboard(image, dx, dy, center_pattern, tiled)
  elif draw_type == ord('3'):
    create_image.cones(image, dx, dy, center_pattern, tiled)
  elif draw_type == ord('4'):
    create_image.helices(image, dx, dy, center_pattern, tiled)
  elif draw_type == ord('5'):
    create_image.spheres(image, dx, dy, center_pattern, tiled)
  elif draw_type == ord('6'):
    create_image.sines(image, dx, dy, center_pattern)
  elif draw_type == ord('7'):
    create_image.gaussian(image, dx, dy, 0.25, center_pattern, tiled)
  else:
    colored = cv2.imread(file_name)
    image = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
  return image

def GetFilter(dx, dy):
  image = np.zeros((512, 512), np.float32)
  create_image.checkerboard(image, dx, dy, True, False)
  #create_image.gaussian(image, dx, dy, 0.25, True, False)
  return image

def ComputeFft(image):
  """Compute FFT of an image"""
  return np.fft.fft2(image)

def ComputeInverseFft(img_fft):
  """Return the module image using the Inverse FFT of an FFT array"""
  inverse = np.fft.ifft2(img_fft)
  img_module = np.absolute(inverse)
  img_module /= np.max(img_module)
  return img_module

def RecenterImage(image):
  """Move the origin from top left to the center and flip the quarters"""
  mid_row = image.shape[0]//2
  mid_col = image.shape[1]//2
  top_left = image[:mid_row,:mid_col]
  bot_left = image[mid_row:,:mid_col]
  top_right = image[:mid_row,mid_col:]
  bot_right = image[mid_row:,mid_col:]
  top = np.concatenate([bot_right,bot_left],axis=1)
  bot = np.concatenate([top_right,top_left],axis=1)
  return np.concatenate([top,bot])

def ExtractModuleAndPhase(img_fft):
  """Convert the FFT into (module,phase)"""
  # Compute module and phase
  img_module = np.absolute(img_fft)
  img_phase = np.angle(img_fft)
  # Rearrange to put the origin in the center
  img_module = RecenterImage(img_module)
  img_phase = RecenterImage(img_phase)
  return img_module, img_phase

def CombineModulePhase(module, phase):
  """Recombine module and phase to build a complex FFT array"""
  # Combine using the Euler formula
  img_fft = module*np.exp(phase*1j)
  # Split the quarters as required by the numpy FFT convention
  img_fft = RecenterImage(img_fft)
  return img_fft

def ConvertModulePhasetoBGR(module, phase):
  # Renormalize the module in [0,1]
  module = np.log(module+1)
  module /= np.max(module)
  # Renormalize the phase in [0,1]
  phase += math.pi
  phase /= 2*math.pi
  # Init the BGR image representation
  width = module.shape[0]
  height = module.shape[1]
  image = np.zeros((width, height, 3), np.uint8)
  # Iterate through the image
  for row in range(height):
    for col in range(width):
      # Map the phase to Hue (number in [0,179] in cv2)
      image[row,col,0] = np.uint8(phase[row,col]*179)
      # Map the module to follow the HSL (instead of HSV) external line:
      # * from black to value = 255 between 0 and 0.5
      # * from saturated to white between 0.5 and 1.0
      if module[row,col] < 0.5:
        image[row,col,1] = 255
        image[row,col,2] = np.uint8(module[row,col]*511)
      else:
        image[row,col,1] = np.uint8((1.0-module[row,col])*511)
        image[row,col,2] = 255
  return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

if __name__ == "__main__":
  help_message = """Create simple B&W test images:
1: Gradient
2: Checkerboard
3: Cones
4: Helices
5: Spheres
6: Sines
7: Gaussians
8: Image (path hardcoded in code, image converted to grayscale)
w, a, s, d: Control the x,y position of the P0 pattern origin point
W, A, S, D: Control the x,y position of the P1 pattern origin point
i: Save the colored FFT image to disk. The name will be auto generated as FFT_<number>.png
c: Toggle center pattern / don't center pattern
t: Toggle tile / don't tile image
f: Toggle filter on phase
m: Toggle filter on module
g: Decrease size of filter
G: Increase size of filter
r: Reset P0 to (0, 0), P1 to (width, width) and tiles to 1
h: Display this message
q: Quit the program"""
  do_something = True
  # Init image generation parameters
  x0 = y0 = 10
  x1 = y1 = 500
  increment = 10
  dx = dy = 10
  draw_type = ord('2')
  center_pattern = True
  tiled = True
  filter_size = 500
  filter_phase = False
  filter_module = False
  save_count = 0
  # Loop till the user hits 'q'
  while do_something:
    # Load or compute an image (grayscale)
    image = GetImage(draw_type, (x0,y0), (x1,y1), dx, dy, center_pattern, tiled, "flower.jpg")
    # Compute its FFT
    img_fft = ComputeFft(image)
    # Compute the module and phase
    module, phase = ExtractModuleAndPhase(img_fft)
    # Filter the module and phase separately (or together or not at all)
    filter = GetFilter(filter_size, filter_size)
    if filter_phase:
      phase = phase * filter
    if filter_module:
      module = module * filter
    # Recombine the module and phase
    img_fft = CombineModulePhase(module, phase)
    # Inverse the FFT into an image
    inverse = ComputeInverseFft(img_fft)
    # Prepare for display the filtered module and phase
    fft_combi = ConvertModulePhasetoBGR(module, phase)
    # Show the images
    cv2.imshow('image',image)
    cv2.imshow('filtered module',module)
    cv2.imshow('filtered phase',phase)
    cv2.imshow('fft',fft_combi)
    cv2.imshow('inverse',inverse)
    # Wait for key stroke and interpert the key
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
      do_something = False
    elif ord('1') <= key <= ord('8'):
      draw_type = key
    elif key == ord('w'):
      y0 -= increment
      dy += 1
    elif key == ord('a'):
      x0 -= increment
      dx = max(1, dx-1)
    elif key == ord('s'):
      y0 += increment
      dy = max(1, dy-1)
    elif key == ord('d'):
      x0 += increment
      dx += 1
    elif key == ord('W'):
      y1 -= increment
    elif key == ord('A'):
      x1 -= increment
    elif key == ord('S'):
      y1 += increment
    elif key == ord('D'):
      x1 += increment
    elif key == ord('r'):
      x0 = y0 = 0
      x1 = y1 = 512
      dx = dy = 10
    elif key == ord('c'):
      center_pattern = not center_pattern
    elif key == ord('t'):
      tiled = not tiled
    elif key == ord('f'):
      filter_phase = not filter_phase
    elif key == ord('m'):
      filter_module = not filter_module
    elif key == ord('g'):
      filter_size -= 10
    elif key == ord('G'):
      filter_size += 10
    elif key == ord('i'):
      # Save the colorized FFT
      name = 'FFT_{0:d}.png'.format(save_count)
      print('Save FFT image : ', name)
      cv2.imwrite(name,fft_combi)
      save_count += 1
    elif key == ord('h'):
      print(help_message)

  cv2.destroyAllWindows()

