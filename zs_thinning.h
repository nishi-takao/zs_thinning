// zs_thinging.h - Zhang-Suen Thinnning Algorithm for OpenCV
//
// Time-stamp: <2017-06-23 09:37:02 zophos>
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
#ifndef __VVV_ZHANG_SUEN_THINNING__
#define __VVV_ZHANG_SUEN_THINNING__
#include <opencv2/core/core.hpp>

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
int zs_thinning(cv::Mat &src,cv::Mat &dst,T bgColor=0);


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
int zs_thinning(cv::Mat &src,T bgColor=0);

//////////////////////////////////////////////////////////////////////////
//
// Following explicit instantnces are defined, those are;
//   single channel: char, uchar, short, ushort, long, ulong, float, double
//   multi channles: cv::Vec3b, cv::Vec3s, cv::Vec3i, cv::Vec3f, cv::Vec3d
//
// If you need another type instantnce, add it to tail of zhang_thinning.cc.
//
#endif
