#!/usr/bin/env bash
nvcc -c -o conv.o conv.cu
g++ -c -I/usr/include/opencv4 -o main.o main.cpp
g++ -o conv main.o conv.o -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_imgproc -lcudart
