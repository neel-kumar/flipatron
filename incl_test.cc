#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
using namespace std;

signed main() {
	// Read the image file as
	Mat image = imread("ferrari.jpg"); // , IMREAD_GRAYSCALE);
	cout << "imread works, include works" << endl;
	return 0;
}
