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

  while (true) {
    clock_t pipeline_time = clock();
    cap >> img;
    if (img.empty()) {
      break;
    }
    //cv::GaussianBlur(img, img_conv, cv::Size(3, 3), 0);
    float run_ms = conv2d(img.data, img_conv.data, rows, cols, kernel);
    cv::imshow("img", img);
    cv::imshow("img_conv", img_conv);
    pipeline_time = clock() - pipeline_time;
    printf("Kernel FPS: %f, Pipeline FPS: %f\n", 1000 / run_ms, 1 / ((float)pipeline_time / CLOCKS_PER_SEC));
    if (cv::waitKey(30) == 'q') {
      break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
