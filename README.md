# README

In this project, denoising experiments mentioned in the paper are included. Here we provide the test code for four datasets. The following is a detailed description

## Platform

1. python==3.9.7
2. tensorflow==2.8.0
3. keras==2.8.0

Currently we can only guarantee that this demo can run normally under Linux.

## Project Structure

- `./data` Include the images of four datasets, namely the BSD-100 (100 images), Kodak-24 (24 images), Urban-100 (100 images) and Set-12（12 images）
- `./model` Include the model parameter file (.h5) which already trained
- `./src` Include scripts including data processing methods and other tool functions

## Instructions

1. Pre-processing: Before running the program, please run the following command to pre-process the project

  ```bash
  python -W ignore main.py --preprocess --dataset=BSD
  python -W ignore main.py --preprocess --dataset=Kodak 
  python -W ignore main.py --preprocess --dataset=Urban 
  ```

2. Showing test image

  We provide tests for four datasets, BSD-100, Kodak-24, Urban-100 and Set-12 which include four different noise levels of `[10, 20, 30, 40]`. For example, if you want to test the denoising result of No.003 image with `sigma=20` in BSD-100, you can run the following command

  ```bash
  python -W ignore main.py --dataset=BSD --test_model --sigma=20 --test_pic_num=003 --pic_show
  ```

3. Saving test image

  E.g.

  ```bash
  python -W ignore main.py --dataset=BSD --test_model --sigma=20 --test_pic_num=2 --pic_save
  ```

## Denoising demo

Here are denoising demos in `./example`, they are "004.png" from BSD-100, "kodim20.png" from Kodak-24 and "035.png" form Urban-100. 
