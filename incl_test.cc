#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
using namespace std;

signed main() {
	// Read the image file as
	Mat image = imread("data/ferrari.jpg"); // , IMREAD_GRAYSCALE);
	Mat resized_down;
	resize(image, resized_down, Size(300, 200), INTER_LINEAR);
	cout << "include, imread, resize WORKS" << endl;
	return 0;
}
