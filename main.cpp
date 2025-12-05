#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>

#include "conv.h"

int main(void) {
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "failed to open default webcam" << std::endl;
    return -1;
  }

  const int rows = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  const int cols = cap.get(cv::CAP_PROP_FRAME_WIDTH);

  std::cout << cols << "x" << rows << std::endl;

  cv::Mat img;
  cv::Mat img_conv(cv::Size(cols, rows), 16);

  const char kernel[KERNEL_ROWS * KERNEL_COLS] = {
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
  };

  bool show_conv = true;
  while (true) {
    clock_t pipeline_time = clock();
    cap >> img;
    if (img.empty()) {
      break;
    }
    //cv::GaussianBlur(img, img_conv, cv::Size(3, 3), 0);
    float run_ms = conv2d(img.data, img_conv.data, rows, cols, kernel);
    if (show_conv) {
      cv::imshow("conv", img_conv);
    } else {
      cv::imshow("conv", img);
    }
    pipeline_time = clock() - pipeline_time;
    printf("Kernel FPS: %f, Pipeline FPS: %f\n", 1000 / run_ms, 1 / ((float)pipeline_time / CLOCKS_PER_SEC));
    int key = cv::waitKey(30);
    if (key == 'q') {
      break;
    } else if (key == ' ') {
      show_conv = !show_conv;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
