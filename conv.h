#define KERNEL_ROWS 3
#define KERNEL_COLS 3

/*
 * Applies 2D convolution to a 3-channel BGR image from OpenCV.
 *
 * Returns kernel execution time in milliseconds.
 */
extern "C" float conv2d(const unsigned char* input, unsigned char* output, int rows, int cols, const char* kernel);
