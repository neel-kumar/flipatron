#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo/cuda.hpp>
#include <iostream>

using namespace std;
using namespace cv;

signed main() {
	Mat img = imread("ferrari.jpg"); // , IMREAD_GRAYSCALE);
	Mat rsize, denoise, threshold, houghline, morph, canny;
	resize(img, rsize, Size(1200, 800), INTER_LINEAR);
	cv::cuda::fastNlMeansDenoising(img, denoise, 30.0, 7, 21);

	// Error Handling
	if (img.empty()) {
		cout << "Image File Not Found" << endl;
		cin.get();
		return -1;
	}

	imshow("", rsize);
	imshow("", denoise);

	while(true) {
		char key = (char) cv::waitKey(30);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed.
	}
}
