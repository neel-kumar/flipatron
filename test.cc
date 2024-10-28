// C++ program for the above approach
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Driver code
int main(int argc, char** argv)
{
	// Read the image file as
	Mat image = imread("data/ferrari.jpg"); // , IMREAD_GRAYSCALE);
	Mat resized_down, blur3x3;
	resize(image, resized_down, Size(300, 200), INTER_LINEAR);
	GaussianBlur(resized_down, blur3x3, Size(3, 3), 0);

	// Error Handling
	if (image.empty()) {
		cout << "Image File "
			<< "Not Found" << endl;

		// wait for any key press
		cin.get();
		return -1;
	}

	// Show Image inside a window with
	// the name provided
	imshow("300x200", resized_down);
	imshow("Gangsta blur", blur3x3);

	// Wait for any keystroke
	while(true) {
		char key = (char) cv::waitKey(30);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed.
	}

	return 0;
}
