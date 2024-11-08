# fft
**Experimentations with FFT on images: computation, visualisation and inverseFFT**

The purpose of this project is to experiment with 2D Fourier Transforms. It comes with a simple way to create test patterns, play with FFT and a more ambitious "colored FFT" program that explores the beauty of phase + module combinations.

**Prerequisite:** The code uses the OpenCV (cv2) library to display images and provides a simple keystroke driven user interaction. The FFT is implemented using the NumPy fft module. NumPy is also used to move pixel arrays around.


The 3 modules can be run independently and double as libraries:

- create_image.py: provides 7 synthetic image primitives to create test images: gradients, checkerboards, cones, sines, gaussians, helices, spheres. All can repeat their pattern as tiles to cover the image. All images are 1 channel grayscale over [0, 255].
- fft_experiments.py: provides the primitives to compute the FFT and FFT inverse. It also extracts the phase and module and recombine them. It also provides an original way to combine the phase and module in a single colored FFT representation. This module uses create_image to test various patterns on the fly without having to create images in advance
- FFX.py: a more playful program allowing to explore colored FFT images. It uses both create_image and fft_experiments

All modules can be launched independently. They are controled with a simple keystroke user interface that provides ways to change the parameters and save the produced images. Hit 'h' to get a help string in the terminal, 'q' to quit the program.

This code is experimental and intended to be used by programmers who want to edit the code for further FFT experiments.


**fft_experiments**

When run independently, this module allows you to experiment with a visualisation of phase, module and inverse FFT. You can filter the phase and module and see the resulting degradation on the inverse image. This is the only module that allows the loading of an image. Note that the image must have width and height that are power of 2 (the module doesn't do padding).

![Filtered FFT](https://github.com/pbossut/fft/blob/main/filtered_fft.png)


**FFX**

This module is more for fun and exploration. You can test the various test pattern and change the arguments a bit to create interesting colored FFT visualisation.

![Colored FFT](https://github.com/pbossut/fft/blob/main/colored_fft.png)

The coloring algorithm is my own as I haven't seen anything like this anywhere. Most FFT tutorial plot module only or module and phase separated. The idea is that, since the phase is an angle, to use it to encode the Hue in an HSV color scheme. The Saturation and Value are computed from the module by following the saturated line (S = 1.0) and change the value till module = 0.5, then decrease the saturation to 0 keeping V = 1.0 for module going from 0.5 to 1.0. This creates pleasing colors when using regular patterns.