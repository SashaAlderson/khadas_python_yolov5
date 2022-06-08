#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <vector>

#ifdef _MSC_VER  
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <string>

#include <pthread.h>
#include <sys/time.h>
#include <sched.h>
#include <sys/resource.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>

#include "common.h"
#include "tengine/c_api.h"

extern "C" {

typedef struct
{
	float x, y, w, h;
} box;

typedef struct
{
	box bbox;
	int classes;
	float* prob;
	float* mask;
	float objectness;
	int sort_class;
} detection;

typedef struct layer
{
	int batch;
	int total;
	int n, c, h, w;
	int out_n, out_c, out_h, out_w;
	int classes;
	int inputs;
	int outputs;
	int* mask;
	float* anchors;
	float* output;
	int coords;
} layer;

void run_graph_wrapper(int64_t p);

void postpress_graph_image_wrapper(int height, int width, float* array, graph_t graph, int output_node_num,
								   int net_w, int net_h, int classes, int num_dets, float nms, float thresh);
									 
int set_graph(int net_h, int net_w, graph_t graph);

int set_image_wrapper(void* data_pointer, int height, int width, tensor_t input_tensor, int net_h, int net_w);

int get_classes(graph_t graph);
}