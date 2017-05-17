#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

void help();
int input_handling(int *, int *, int, char**);
double average_fps( double *array_fps, double *fps_mat, double *fps_frame, 
					double *fps_cvtc, double *fps_gblur, double *fps_sobel,
					double *fps_sobel_cvt, double *fps_sobel2, double *fps_sobel_cvt2,
					double *fps_add_weighted, double *fps_imshow, int size);

int main(int argc, char** argv )
{
	int videoSrc, nsamples=1000;
	cout << "OpenCV version: " << CV_MAJOR_VERSION << '.' << CV_MINOR_VERSION << '\n';

	help();
	if (input_handling(&videoSrc, &nsamples, argc, argv) < 0) return -1;

	//VideoCapture cap(0); // open the default camera
	/*VideoCapture cap(videoSrc); // open the device passed as argument
    if(!cap.isOpened()){  // check if we succeeded
		cout << "Video device could not be opened\n";
        return -1;
	}*/

    namedWindow("edges",1);

	double	t_ini, t_mat, t_cap, t_cvtc, t_gblur, t_sobel, t_sobel_cvt,
			t_add_weighted, t_imshow, t_end, t_sobel2;
	double *fps = new double [nsamples];
	double *fps_mat = new double [nsamples];
	double *fps_frame = new double [nsamples];
	double *fps_cvtc = new double [nsamples];
	double *fps_gblur = new double [nsamples];
	double *fps_sobel = new double [nsamples];
	double *fps_sobel_cvt = new double [nsamples];
	double *fps_sobel2 = new double [nsamples];
	double *fps_sobel_cvt2 = new double [nsamples];
	double *fps_add_weighted = new double [nsamples];
	double *fps_imshow = new double [nsamples];

	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	//Mat frame;
	Mat edges;
	
    for(unsigned int i = 0; i < (nsamples-1); i++)
    {
		t_ini = (double)getTickCount();
		Mat frame;
		t_mat = (double)getTickCount();
		//cap >> frame; // get a new frame from camera
		frame = imread("../image.jpg", 1);

		t_cap = (double)getTickCount();
		cvtColor(frame, edges, COLOR_BGR2GRAY);
		t_cvtc = (double)getTickCount();		
		GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        
		t_gblur = (double)getTickCount();
		/// Gradient X
		Sobel( edges, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
		t_sobel = (double)getTickCount();
		convertScaleAbs( grad_x, abs_grad_x );
		t_sobel_cvt = (double)getTickCount();

		/// Gradient Y
		Sobel( edges, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
		t_sobel2 = (double)getTickCount();
		convertScaleAbs( grad_y, abs_grad_y );

		t_add_weighted = (double)getTickCount();
		/// Total Gradient (approximate)
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges );

		t_imshow = (double)getTickCount();
		imshow("edges", edges);
		t_end = (double)getTickCount();
		if(waitKey(1) >= 0) break;

		fps[i] = ((t_end - t_ini)/getTickFrequency());
		fps_mat[i] = ((t_mat - t_ini)/getTickFrequency());
		fps_frame[i] = ((t_cap - t_mat)/getTickFrequency());
		fps_cvtc[i] = ((t_cvtc - t_cap)/getTickFrequency());
		fps_gblur[i] = ((t_gblur - t_cvtc)/getTickFrequency());
		fps_sobel[i] = ((t_sobel - t_gblur)/getTickFrequency());
		fps_sobel_cvt[i] = ((t_sobel_cvt - t_sobel)/getTickFrequency());
		fps_sobel2[i] = ((t_sobel2 - t_sobel_cvt)/getTickFrequency());
		fps_sobel_cvt2[i] = ((t_add_weighted - t_sobel2)/getTickFrequency());
		fps_add_weighted[i] = ((t_imshow - t_add_weighted)/getTickFrequency());
		fps_imshow[i] = ((t_end - t_imshow)/getTickFrequency());		
    }

	average_fps(fps, fps_mat, fps_frame, fps_cvtc, fps_gblur, fps_sobel,
				fps_sobel_cvt, fps_sobel2, fps_sobel_cvt2, fps_add_weighted,
				fps_imshow, nsamples);
	delete fps;
    return 0;
}

void help()
{
	cout << "Usage: ./SobelDerivatives <input-number> [<samples>]\n";
	cout << "input-number - is the video device number. Defaults to zero.\n";
	cout << "samples - number of samples. Defaults to 1000; zero means infinite.\n";
}

int input_handling(int *src, int* nsmp, int argc, char** argv)
{
	if(argc < 2){
		cerr << "Video device number required, e.g. for /dev/video2 just use '2'\n";
		help();
		return -1;
	} else if(argc == 2){
		// convert argument to integer
		istringstream ss(argv[1]); 
		*src;
		if(!(ss >> *src)){
			cerr << "Invalid number " << argv[1] << '\n';
			return -1;
		}
	} else if(argc == 3){
		// convert argument to integer
		istringstream ss1(argv[1]); 
		*src;
		if(!(ss1 >> *src)){
			cerr << "Invalid number " << argv[1] << '\n';
			return -1;
		}

		// convert argument to integer
		istringstream ss2(argv[2]); 
		*nsmp;
		if(!(ss2 >> *nsmp)){
			cerr << "Invalid number " << argv[1] << '\n';
			return -1;
		}
	} else{

		cerr << "Too many parameters";
		help();
		return -1;
	}

	return 0;
}

double average_fps(	double *array_fps, double *fps_mat, double *fps_frame, 
					double *fps_cvtc, double *fps_gblur, double *fps_sobel,
					double *fps_sobel_cvt, double *fps_sobel2, double *fps_sobel_cvt2,
					double *fps_add_weighted, double *fps_imshow, int size)
{
	double	t_ini = 0.0, t_mat = 0.0, t_frame = 0.0, t_cvtc = 0.0, t_gblur = 0.0,
			t_sobel = 0.0, t_sobel_cvt = 0.0, t_sobel2 = 0.0, t_add_weighted = 0.0,
			t_imshow = 0.0, t_end = 0.0;
	double elapsed = 0.0;

	cout << "sample_number,elapsed_time,frame,mat_alloc,read_img,";
	cout << "convert_color,gaussian_blur,sobel,convert_scale,sobel2,";
	cout << "conver_scale2,add_weighted,show_img\n";

	cout.precision(7);
	for(unsigned int i = 0; i < size; i++)
    {
		//Print CSV like so just need to redirect stdout to file
		t_ini += array_fps[i];
		t_mat += fps_mat[i];
		t_frame += fps_frame[i];
		t_cvtc += fps_cvtc[i];
		t_gblur += fps_gblur[i];
		t_sobel += fps_sobel[i];
		t_sobel_cvt += fps_sobel_cvt[i];
		t_sobel2 += fps_sobel2[i];
		t_add_weighted += fps_sobel_cvt2[i];
		t_imshow += fps_add_weighted[i];
		t_end += fps_imshow[i];

		cout << i << ',' << 
				elapsed << ',' << 
				array_fps[i] << ',' << 
				fps_mat[i] << ',' << 
				fps_frame[i] << ',' << 
				fps_cvtc[i] << ',' << 
				fps_gblur[i] << ',' << 
				fps_sobel[i] << ',' << 
				fps_sobel_cvt[i] << ',' <<
				fps_sobel2[i] << ',' << 
				fps_sobel_cvt2[i] << ',' << 
				fps_add_weighted[i] << ',' << 
				fps_imshow[i] << '\n';

		elapsed += array_fps[i];
	}
	cout << "\nAverages:\n";
	cout << t_ini/size << ',' << 
			t_mat/size << ',' << 
			t_frame/size << ',' << 
			t_cvtc/size << ',' << 
			t_gblur/size << ',' << 
			t_sobel/size << ',' << 
			t_sobel_cvt/size << ',' << 
			t_sobel2/size << ',' << 
			t_add_weighted/size << ',' << 
			t_imshow/size << ',' <<
			t_end/size << '\n';

	return t_ini/size;
}
