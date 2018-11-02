/*
 * main.cpp
 *
 *  Created on: Nov 26, 2017
 *      Author: deepgray
 */

#include"layer.h"
#include<cstdlib>
#include<iostream>
#include"util.h"

extern "C" int deviceQuery();

int main()
{
	Net n;
	Net* p = &n;
	//deviceQuery();
	netInit(p);

	for(int i = 0; i < 100; ++i)
	{
		//std::cout<<"steps "<<i<<" computing..\n";
		forward(p);
	}

	netDestroy(p);

	return 0;
}
