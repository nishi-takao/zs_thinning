// zs_thinging.cc - Zhang-Suen Thinnning Algorithm for OpenCV
//
// Time-stamp: <2017-06-23 13:38:38 zophos>
//
// based on ImageJ BinaryProcessor.java
// https://imagej.nih.gov/ij/source/ij/process/BinaryProcessor.java
//
// T. Y. Zhang and C. Y. Suen
// A Fast Parallel Algorithm for Thinning Digital Patterns 
// CACM 27(3):236--239, 1984
// http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
//
//
//  Copyright (C) 2017  NISHI, Takao <zophos@ni.aist.go.jp>
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met: 
//  * Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation 
//    and/or other materials provided with the distribution.
//  * Neither the name of the AIST nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL NISHI, Takao BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//
#include <opencv2/core/core.hpp>


////////////////////////////////////////////////////////////////////////
//
// apply thinning for each pixel
//
// cv::Mat &src: source matrix (1ch)
//               SHOULD have 1px padding for each edge
// cv::Mat &dst: destination maxrix;
//               SHOLD have same properties with src,
//               but independent of src.
// int stage:    stage number (0-origin, 0 or non-0)
// int pass:     pass number (0-origin)
// T bgColor:    background color
//
// return: number of removed pixels
//
template <class T>
static int _thin(cv::Mat &src,
                 cv::Mat &dst,
                 int stage,
                 int pass,
                 T bgColor)
{
    //
    // Uses a lookup table to repeatably removes pixels from the
    // edges of objects in a binary image, reducing them to single
    // pixel wide skeletons. There is an entry in the table for each
    // of the 256 possible 3x3 neighborhood configurations. An entry
    // of '1' means delete pixel on first pass, '2' means delete pixel on
    // second pass, and '3' means delete on either pass. Pixels are
    // removed from the right and bottom edges of objects on the first
    // pass and from the left and top edges on the second pass.
    // A graphical representation of the 256 neighborhoods indexed by
    // the table is available at
    // "http://imagej.nih.gov/ij/images/skeletonize-table.gif".
    //
    static const uchar REMOVE_FLAGS[][256]={
        {
            0,0,0,0,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,0,0,2,0,3,0,3,3,
            0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
            0,0,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0
        },
        {
            0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,2,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        }
    };

    //
    // choose remove flag table and mask by stage and pass number
    //
    const uchar *rflags=REMOVE_FLAGS[stage?1:0]; // 1st.: 0 / 2nd.: 1
    const uchar rmask=(pass&1)?2:1; // even (1st.): 1 / odd (2nd.): 2

    //
    // clear dst buffer with bgcolor
    //
    dst.setTo(bgColor);

    const int y_end=src.rows-1;
    const int x_end=src.cols-1;

    //
    // neighborhood name:
    //
    //   p1 | p2 | p3
    //  ----+----+----
    //   p4 | p5 | p6
    //  ----+----+----
    //   p7 | p8 | p9
    //
    //
    // index bit assignment:
    //
    //  0x01|0x02|0x04
    //  ----+----+----
    //  0x80| ** |0x08
    //  ----+----+----
    //  0x40|0x20|0x10
    //
    //  ** p5 to 0x100 for convenience
    //
    int pixelsRemoved=0;
    for(int py=0,cy=1,ny=2;ny<y_end;py++,cy++,ny++){

        //
        // create initial index
        // don't care about a left row (p1, p4, p7)
        //
        int cx=0,nx=1;
        int index=0;
        if(src.at<T>(py,cx)!=bgColor) // p2
            index|=0x02;
        if(src.at<T>(py,nx)!=bgColor) // p3
            index|=0x04;

        T p5=src.at<T>(cy,cx);
        if(p5!=bgColor)
            index|=0x100;
        T p6=src.at<T>(cy,nx);
        if(p6!=bgColor)
            index|=0x08;

        if(src.at<T>(ny,cx)!=bgColor) // p8
            index|=0x20;
        if(src.at<T>(ny,nx)!=bgColor) // p9
            index|=0x10;

        for(++cx,++nx;nx<x_end;cx++,nx++){
            //
            // slide the index
            //
            // p2->p1 (0x02- > 0x01), p3->p2 (0x04 -> 0x02),
            // p5->p4 (0x100-> 0x80), p6->p5 (don't care)
            // p8->p7 (0x20 -> 0x40), p9->p8 (0x10 -> 0x20)
            //
            index=((index&0x106)>>1)|((index&0x030)<<1);
            p5=p6;

            //
            // update the index for a new row (p3, p6, p9)
            //
            if(src.at<T>(py,nx)!=bgColor) // p3
                index|=0x04;

            p6=src.at<T>(cy,nx);
            if(p6!=bgColor)
                index|=0x08;

            if(src.at<T>(ny,nx)!=bgColor) // p9
                index|=0x10;


            if(p5!=bgColor){
                if(rflags[index]&rmask)
                    pixelsRemoved++;
                else
                    dst.at<T>(cy,cx)=p5;

                //
                // set p5 bit to the index for next window
                //
                index|=0x100;
            }
        }
        //
        // end of col-loop
        //
    }


    return pixelsRemoved;
}


////////////////////////////////////////////////////////////////////////
//
// Zhang-Suen's thinning
//
// cv::Mat &src:  source matrix
// cv::Mat &dst:  destination maxrix
// T bgColor: background color (defualt: 0)
//
// return: number of iteration
//
template <class T>
int zs_thinning(cv::Mat &src,cv::Mat &dst,T bgColor=0)
{
    //
    // create two working buffers, those have 1px padding for each edge
    // to guard overrun.
    //
    cv::Mat work0(src.rows+2,src.cols+2,
                  CV_MAKETYPE(src.depth(),src.channels()));
    work0.setTo(bgColor);
    cv::Rect roi=cv::Rect(1,1,src.cols,src.rows);
    cv::Mat win=work0(roi);
    src.copyTo(win);

    cv::Mat work1=work0.clone();


    int pass=0,pixelsRemoved=0;

    //
    // 1st. stage (stage-0)
    //
    do{
        pixelsRemoved=_thin<T>(work0,work1,0,pass++,bgColor);
        pixelsRemoved+=_thin<T>(work1,work0,0,pass++,bgColor);
    }while(pixelsRemoved>0);

    //
    // 2nd. stage: remove "stuck" pixels  (stage-1)
    //
    do{
        pixelsRemoved=_thin<T>(work0,work1,1,pass++,bgColor);
        pixelsRemoved+=_thin<T>(work1,work0,1,pass++,bgColor);
    }while(pixelsRemoved>0);


    win.copyTo(dst);

    return pass;
}

////////////////////////////////////////////////////////////////////////
//
// Zhang-Suen's thinning (overwrite version)
//
// cv::Mat &src:  source matrix; it will be overwritten.
// T bgColor: background color (defualt: 0)
//
// return: number of iteration
//
template <class T>
int zs_thinning(cv::Mat &src,T bgColor=0)
{
    return zs_thinning<T>(src,src,bgColor);
}

//////////////////////////////////////////////////////////////////////////
//
// Explicit instantiation
// single channel:
//     char, uchar, short, ushort, long, ulong, float, double
// multi channels:
//     cv::Vec3b, cv::Vec3s, cv::Vec3i, cv::Vec3f, cv::Vec3d
//
template
int zs_thinning<char>(cv::Mat &src,char bgColor);
template
int zs_thinning<char>(cv::Mat &src,cv::Mat &dst,char bgColor);

template
int zs_thinning<uchar>(cv::Mat &src,uchar bgColor);
template
int zs_thinning<uchar>(cv::Mat &src,cv::Mat &dst,uchar bgColor);

template
int zs_thinning<short>(cv::Mat &src,short bgColor);
template
int zs_thinning<short>(cv::Mat &src,cv::Mat &dst,short bgColor);

template
int zs_thinning<ushort>(cv::Mat &src,ushort bgColor);
template
int zs_thinning<ushort>(cv::Mat &src,cv::Mat &dst,ushort bgColor);

template
int zs_thinning<long>(cv::Mat &src,long bgColor);
template
int zs_thinning<long>(cv::Mat &src,cv::Mat &dst,long bgColor);

template
int zs_thinning<ulong>(cv::Mat &src,ulong bgColor);
template
int zs_thinning<ulong>(cv::Mat &src,cv::Mat &dst,ulong bgColor);

template
int zs_thinning<float>(cv::Mat &src,float bgColor);
template
int zs_thinning<float>(cv::Mat &src,cv::Mat &dst,float bgColor);

template
int zs_thinning<double>(cv::Mat &src,double bgColor);
template
int zs_thinning<double>(cv::Mat &src,cv::Mat &dst,double bgColor);

template
int zs_thinning<cv::Vec3b>(cv::Mat &src,cv::Vec3b bgColor);
template
int zs_thinning<cv::Vec3b>(cv::Mat &src,cv::Mat &dst,cv::Vec3b bgColor);

template
int zs_thinning<cv::Vec3s>(cv::Mat &src,cv::Vec3s bgColor);
template
int zs_thinning<cv::Vec3s>(cv::Mat &src,cv::Mat &dst,cv::Vec3s bgColor);

template
int zs_thinning<cv::Vec3i>(cv::Mat &src,cv::Vec3i bgColor);
template
int zs_thinning<cv::Vec3i>(cv::Mat &src,cv::Mat &dst,cv::Vec3i bgColor);

template
int zs_thinning<cv::Vec3f>(cv::Mat &src,cv::Vec3f bgColor);
template
int zs_thinning<cv::Vec3f>(cv::Mat &src,cv::Mat &dst,cv::Vec3f bgColor);

template
int zs_thinning<cv::Vec3d>(cv::Mat &src,cv::Vec3d bgColor);
template
int zs_thinning<cv::Vec3d>(cv::Mat &src,cv::Mat &dst,cv::Vec3d bgColor);

//
// add your types if you need.
//
