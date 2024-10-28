#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace std;
using namespace cv;

// https://gist.github.com/itsrifat/66b253db2736b23f247c
/*
This functions opens a video file and extracts the frames and put them into a vector of Mat(its the class for representing an img)
*/
void extract_frames(const string &videoFilePath,vector<Mat>& frames){

	try{
		//open the video file
  	VideoCapture cap(videoFilePath); // open the video file
  	if(!cap.isOpened()) { // check if we succeeded
  		CV_Error(Error::StsError, "Can not open Video file");
      std::cout << "Error opening video file" << endl;
    }

  	//cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
  	for(int frameNum = 0; frameNum < cap.get(CAP_PROP_FRAME_COUNT); frameNum++)
  	{
  		Mat frame;
  		cap >> frame; // get the next frame from video
  		frames.push_back(frame);
  	}
  }
  catch( cv::Exception& e ){
    cerr << e.msg << endl;
    exit(1);
  }

}

/*
It saves a vector of frames into jpg images into the outputDir as 1.jpg,2.jpg etc where 1,2 etc represents the frame number
*/
void save_frames(vector<Mat>& frames, const string& outputDir){
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

  long long frameNumber = 0;

	for(std::vector<Mat>::iterator frame = frames.begin(); frame != frames.end(); ++frame, ++frameNumber) {
	  string filePath = outputDir + to_string(static_cast<long long>(frameNumber))+ ".jpg";
          cout << "Storing " << frameNumber << " th frame to file: " << filePath << endl;
	  imwrite(filePath,*frame,compression_params);
	}


}

signed main(int argc, char** argv)
{
    std::vector<Mat> frames;
    if (argc > 1) {
        std::cout << "Opening file " << argv[1] << endl;
        extract_frames(argv[1], frames);
    }

    if (argc > 2) {
        std::cout << "Saving frames to " << argv[2] << endl;
        save_frames(frames, argv[2]);
    }

    return 0;
}

