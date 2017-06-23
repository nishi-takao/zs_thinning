// test-zs_thinning.cc -- sample program for zs_thinning
//
// Time-stamp: <2017-06-23 09:14:12 zophos>
//
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "zs_thinning.h"

int main(int argc,char *argv[])
{
    if(argc<2){
        std::cerr<<argv[0]<<" filename [outfile]"<<std::endl;
        exit(-1);
    }

    //
    // read as grayscale image
    //
    cv::Mat src=cv::imread(argv[1],0);
    cv::Mat dst;

    //
    // do thinning
    //
    zs_thinning<uchar>(src,dst);

    //
    // show result
    //
    cv::namedWindow("result");
    cv::imshow("result",dst);
    cv::waitKey(0);

    //
    // save result image
    //
    if(argc>2)
        cv::imwrite(argv[2],dst);
    
    return 0;
}
