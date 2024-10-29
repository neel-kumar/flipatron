// https://github.com/umerghafoor/Documnet-Scanner/blob/main/Document-Scanner/Source.cpp

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

////////////////////////////////////////////////////////
///////////////  Document Scanner //////////////////////
////////////////////////////////////////////////////////

/* These are variable declarations in C++ for an image processing program. */

/*
cv::Mat img, imggryblr, imgcany;
cv::Mat imgcontr;
cv::Mat imgwarp;
cv::Mat imgclr;
std::vector<cv::Point> imgpoint;
std::vector<cv::Point> newpoints;
float w = 420, h = 596;
*/

/**
 * The function draws a set of points and lines to form a contour on an image.
 * 
 * @param points A vector of cv::Point objects representing the four corners of a rectangle.
 * @param color The color of the lines and circles to be drawn, specified as a cv::Scalar object.
 */
void drawconts(cv::Mat& img, std::vector<cv::Point> points, cv::Scalar color)
{
	for (int i = 0; i < points.size(); i++)
	{
		circle(img, points[i], 4, color, cv::FILLED);
	}
	line(img, points[0], points[1], color, 2);
	line(img, points[1], points[3], color, 2);
	line(img, points[2], points[3], color, 2);
	line(img, points[0], points[2], color, 2);
}

/**
 * The function preprocesses an input image by converting it to grayscale, applying Gaussian blur,
 * Canny edge detection, and dilation.
 * 
 * @param img The input image that needs to be preprocessed.
 * 
 * @return The function `preprocess` returns a `cv::Mat` object, which is the processed image after
 * applying various image processing techniques such as converting to grayscale, Gaussian blur, Canny
 * edge detection, and dilation.
 */
cv::Mat preprocess(cv::Mat img, bool debug)
{
    cv::Mat imggryblr, imgcany;

	cvtColor(img, imggryblr, cv::COLOR_BGR2GRAY);
    if (debug) cv::imshow("cvtColor", imggryblr);
	GaussianBlur(imggryblr, imggryblr, cv::Size(5, 5), 5, 0);
    if (debug) cv::imshow("GaussianBlur", imggryblr);
	Canny(imggryblr, imgcany, 30, 100);
    if (debug) cv::imshow("Canny", imgcany);
	cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	dilate(imgcany, imgcany, k);
    if (debug) cv::imshow("dilate", imgcany);
	return imgcany;
}

void draw_contours(cv::Mat src,
    std::vector<cv::Vec4i>& hierarchy,
	std::vector<std::vector<cv::Point>>& contours,
    int agt)
{
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int nh = 0;
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
    {
        int area = contourArea(contours[idx]);
		if (area > agt) {
            ++nh;
            cv::Scalar color( rand()&255, rand()&255, rand()&255 );
            cv::drawContours( dst, contours, idx, color, cv::FILLED, 8, hierarchy );
        }
    }

    std::cout << "Number of contours: " << contours.size()  << " number of top-level: " << nh << std::endl;

    cv::imshow( "Components", dst );
    cv::waitKey(0);
}

/**
 * The function takes in an image and its canny edge detection output, finds the contours of the image,
 * approximates the contours to polygons, and returns the largest quadrilateral polygon found.
 * 
 * @param imgcany A binary image obtained after applying Canny edge detection on the original image.
 * @param img The input image on which the contours are drawn.
 * 
 * @return A vector of cv::Point representing the four corners of the largest quadrilateral contour in
 * the input image that has an area greater than 10000.
 */
std::tuple<int, std::vector<cv::Point>> getContours(cv::Mat imgcany, int agt = 10000, bool debug=false) {

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hirarchy;

	findContours(imgcany, contours, hirarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (debug) draw_contours(imgcany, hirarchy, contours, agt);

	std::vector<std::vector<cv::Point>> contpoly(contours.size());
	std::vector<cv::Rect> brect(contours.size());

	std::vector<cv::Point> biggest;
	int maxarea = agt;

	for (int i = 0;i < contours.size();i++)
	{
		int area = contourArea(contours[i]);
		if (area > maxarea)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], contpoly[i], 0.02 * peri, true);
			if (contpoly[i].size() == 4)
			{
				biggest = { contpoly[i][0],contpoly[i][1], contpoly[i][2], contpoly[i][3] };
				maxarea = area;
			} else {
                // TODO: Consider boundingRect
            }
			
		}
	}
	return std::make_tuple(maxarea, biggest);
}
/**
 * The function reorders a vector of points based on their x and y coordinates.
 * 
 * @param repoint A vector of cv::Point objects representing the four corners of a quadrilateral.
 * 
 * @return The function `reorder` returns a `std::vector<cv::Point>` which contains the reordered
 * points.
 */
std::vector<cv::Point> reorder(std::vector<cv::Point> repoint)
{
	std::vector<cv::Point> newpoint;
	std::vector<int> sump, subp;

	for (int i = 0;i < 4;i++)
	{
		sump.push_back(repoint[i].x + repoint[i].y);
		subp.push_back(repoint[i].x - repoint[i].y);
	}

	newpoint.push_back(repoint[min_element(sump.begin(), sump.end()) - sump.begin()]);
	newpoint.push_back(repoint[max_element(subp.begin(), subp.end()) - subp.begin()]);
	newpoint.push_back(repoint[min_element(subp.begin(), subp.end()) - subp.begin()]);
	newpoint.push_back(repoint[max_element(sump.begin(), sump.end()) - sump.begin()]);

	return newpoint;
}
/**
 * The function warps an input image based on four given points and returns the warped image with a
 * cropped region.
 * 
 * @param img The input image that needs to be warped.
 * @param point A vector of 4 points representing the corners of a quadrilateral in the input image
 * that needs to be warped.
 * @param w width of the output image after perspective transformation
 * @param h The height of the output image after perspective transformation.
 * 
 * @return a cv::Mat object, which is the warped image after applying perspective transformation and
 * cropping.
 */
cv::Mat warp(cv::Mat img, std::vector<cv::Point> point, float w, float h)
{
	cv::Point2f scr[4] = { point[0],point[1], point[2], point[3] };
	cv::Point2f dst[4] = { {0.0f,0.0f},{w,0.0f}, {0.0f,h}, {w,h} };

	cv::Mat matrix = getPerspectiveTransform(scr, dst);
    cv::Mat imgwarp;
	warpPerspective(img, imgwarp, matrix, cv::Point(w, h));

	float crpv = 5;
	cv::Rect roi(crpv, crpv, w - (2 * crpv), h - (2 * crpv));
	imgwarp = imgwarp(roi);

	return imgwarp;
}

// https://gist.github.com/itsrifat/66b253db2736b23f247c
/*
This functions opens a video file and extracts the frames and put them into a vector of Mat(its the class for representing an img)
*/
void extract_frames(const std::string &videoFilePath, std::vector<cv::Mat> &frames)
{
    try
    {
        // open the video file
        cv::VideoCapture cap(videoFilePath); // open the video file
        if (!cap.isOpened())
        { 
            // check if we succeeded
            CV_Error(cv::Error::StsError, "Can not open Video file");
            std::cout << "Error opening video file" << std::endl;
        }

        // cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
        for (int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
        {
            cv::Mat frame;
            cap >> frame; // get the next frame from video
            frames.push_back(frame);
        }
    }
    catch (cv::Exception &e)
    {
        std::cerr << e.msg << std::endl;
        exit(1);
    }
}

/**
 * Analyze a frame
 * @param img - frame
 * @return - a tuple of area of the largest rectangle, cropped image, and the frame with the rectangle
 *         - area is 0 if no large rectangular contour found in the frame
 */
std::tuple<int, cv::Mat, cv::Mat> afrm(cv::Mat img, bool debug=false)
{   
    const int imgarea = img.size().width * img.size().height;
    std::cout << "Image: width: " << img.size().width << " height: " << img.size().height << " area: " << imgarea << std::endl;
    if (debug) cv::imshow("Original image", img);
    cv::Mat imgcany = preprocess(img, debug);
    if (debug) cv::imshow("Image Canny", imgcany);

    cv::Mat imgwarp, imgwarpflip;

    int maxarea = 0;
    std::vector<cv::Point> imgpoint;
    std::tie(maxarea, imgpoint) = getContours(imgcany, imgarea/3, debug);
    if (debug) cv::imshow("getContours", imgcany);
    if (!imgpoint.empty())
    {
        std::cout << "Found rectangle with maxarea: " << maxarea << std::endl;
        std::vector<cv::Point> newpoints = reorder(imgpoint);

        float w = 420, h = 596;
        imgwarp = warp(img, newpoints, w, h);
        if (debug) cv::imshow("warp", imgwarp);
        flip(imgwarp, imgwarpflip, 1);
        if (debug) imshow("flip warp", imgwarpflip);
        // imwrite("Resources/Scaned/1.png", imgwarp);
        drawconts(img, newpoints, cv::Scalar(0, 255, 255));
        if (debug) imshow("drawconts", img);
    }
    else
    {
        std::cout << "No large contour found!" << std::endl;
        maxarea = 0;
    }

    return std::make_tuple(maxarea, imgwarp, img);
}

/**
 * frames - all the frames in the video in the correct sequence.
 * scans - list of cropped pages in the video
 * rects - rectangle drawn in the frames with the scans
 * odn - output directory name to save the frames
 * 
 * Assumption: User is capturing video and flipping the pages. Every page is rectangular and covers
 *             > 50% of the screen in at least one frame of the video. The video continuously taken
 *             during page flips as well.
 * Algorithm: Iterate through the frames
 *              if there is no large countour, switch to the next page
 *              if there is no current page, store the area, scan and the rectangle
 *              if there is a current page, and current area is large than the previous area
 *                replace the current page's details
 */
void find_scans(std::vector<cv::Mat> &frames, std::vector<int>& fns, std::vector<cv::Mat> &scans, std::vector<cv::Mat> &rects, std::string& odn)
{
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);

    long long fn = 0;
    int cur_area = 0;

    for(cv::Mat& frame: frames) {
        std::cout << "Processing " << fn << " th frame" << std::endl;
        const int num_width = 5;
        std::string numStr = std::to_string(fn);
        numStr.insert(0, num_width - numStr.length(), '0');
        std::string filePath = odn + "/" + numStr + ".jpg";
        std::cout << "Storing " << fn << " th frame to file: " << filePath << std::endl;
	    cv::imwrite(filePath, frame, compression_params);

        int area;
        cv::Mat cr_page, page_rect;
        std::tie(area, cr_page, page_rect) = afrm(frame);
        if (area != 0) {
            if (cur_area == 0) {
                // This is the first page or the first next page
                std::cout << "* Beginning of the next scanned page with cropped area: " << area << std::endl;
                cur_area = area;
                fns.push_back(fn);
                scans.push_back(cr_page);
                rects.push_back(page_rect);
            } else if (area > cur_area) {
                // This is larger than the previous scan of the same page, replace the details
                std::cout << "** Found a better page with cropped area: " << area << std::endl;
                cur_area = area;
                fns[fns.size() - 1] = fn;
                scans[scans.size() - 1] = cr_page;
                rects[rects.size() - 1] = page_rect;
            } else {
                std::cout << "Ignoring, found a worse page with cropped area: " << area << std::endl;
            }
        } else {
            cur_area = 0;
        }

        ++fn;
    }
}

signed debug_frame(std::string ffn)
{
    std::cout << "Reading image: " << ffn << std::endl;
    cv::Mat img = cv::imread(ffn);
    int area;
    cv::Mat cr_page, page_rect;
    std::tie(area, cr_page, page_rect) = afrm(img, true);

    std::cout << "Waiting for a keypress..." << std::endl;
    cv::waitKey(0);

    return 0;
}

signed main(int argc, char** argv)
{
    std::string debug = "-d";
    if (debug.compare(argv[1]) == 0) {
        return debug_frame(argv[2]);
    }

    std::string vfn = argv[1];
    std::string odn = argv[2];

    std::vector<cv::Mat> frames;
    std::cout << "Processing video file: " << vfn << std::endl;

    extract_frames(vfn, frames);
    std::cout << "Number of extracted frames: " << frames.size() << std::endl;

    std::vector<int> fns;
    std::vector<cv::Mat> scans, rects;
    find_scans(frames, fns, scans, rects, odn);

    std::cout << "Total number of pages found: " << scans.size() << std::endl;
    for(int i = 0; i < scans.size(); i++) {
        const int fn = fns[i];
        std::cout << "Page: " << i << " is from frame: " << fn << std::endl;
        std::string label = "cropped page " + std::to_string(fn);
        cv::imshow(label, scans[i]);
        label = "image page " + std::to_string(fn);
        cv::imshow(label, rects[i]);
    }

    std::cout << "Waiting for a keypress..." << std::endl;
    cv::waitKey(0);

    return 0;
}