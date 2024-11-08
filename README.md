# fft
**Experimentations with FFT on images: computation, visualisation and inverseFFT**

The purpose of this project is to experiment with 2D Fourier Transforms. It comes with a simple way to create test patterns, play with FFT and a more ambitious "colored FFT" program that explores the beauty of phase + module combinations.

**Prerequisite:** The code uses the OpenCV (cv2) library to display images and provides a simple keystroke driven user interaction. The FFT is implemented using the NumPy fft module. NumPy is also used to move pixel arrays around.


The 3 modules can be run independently and doubles as libraries:

- create_image.py: provides 7 synthetic image primitives to create test images: gradients, checkerboards, cones, sines, gaussians, helices, spheres. All can repeat their pattern as tiles to cover the image. All images are 1 channel grayscale over [0, 255].
- fft_experiments.py: provides the primitives to compute the FFT and FFT inverse. It also extracts the phase and module and recombine them. It also provides an original way to combine the phase and module in a single colored FFT representation. This module uses create_image to test various patterns on the fly without having to create images in advance
- FFX.py: a more playful program allowing to explore colored FFT images. It uses both create_image and fft_experiments

All modules can be launched independently. They are controled with a simple keystroke user interface that provides ways to change the parameters and save the produced images. Hit 'h' to get a help string in the terminal, 'q' to quit the program.

This code is experimental and intended to be used by programmers who want to edit the code for further FFT experiments.



![flower](https://github.com/pbossut/fft/blob/main/flower.jpg)
