#include "geometry.h"
#include <math.h>

double squared_euclidean128(double x[128], double y[128])
{
	double Sum = 0.0;
	for(int i=0;i<128;i+=4)
	{
		Sum += pow((x[i  ]-y[i  ]), 2.0);
	    Sum += pow((x[i+1]-y[i+1]), 2.0);
	    Sum += pow((x[i+2]-y[i+2]), 2.0);
	    Sum += pow((x[i+3]-y[i+3]), 2.0);
	}
	return Sum;
}

double squared_euclidean2d(double x[2], double y[2])
{
	double diff = x[0]-y[0];
    double sum = diff * diff;
    diff = x[1]-y[1];

	return sum + diff * diff;
}

double squared_euclidean3d(double x[3], double y[3])
{
	double diff = x[0]-y[0];
    double sum = diff * diff;

    diff = x[1]-y[1];
    sum += diff * diff;
    diff = x[2]-y[2];

    return sum + diff * diff;
}