#!/usr/bin/make -f
#
#
OPENCV_LIBS=-lopencv_core -lopencv_highgui

LIBS= -lstdc++ $(OPENCV_LIBS)

CC= g++
CXXFLAGS= -Wall -g -O3

OBJS:=$(patsubst %.cc,%.o,$(wildcard *.cc))
DST:=test-zs_thinning

#.SUFFIXES: .o .cc
#
#.cc.o:
#	$(CC) $(CXXFLAGS) -c $<

all: $(DST)

$(DST): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	-rm $(OBJS) $(DST) *~
