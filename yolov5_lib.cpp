/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "yolov5_lib.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define VXDEVICE  "TIMVX"

#define MODEL_COLS 352
#define MODEL_ROWS 352

#define CAMERA_WEIGHT 640
#define CAMERA_HEIGHT 480

#if DEBUG_OPTION
static unsigned int tmpVal;
#endif
using namespace std;

// const int classes = 80;
//const float thresh = 0.2;
const int numBBoxes = 5;
const int relative = 1;
const int yolov3_numAnchors = 6;
const int yolov2_numAnchors = 5;


int num_thread = DEFAULT_THREAD_COUNT;

 
void get_input_data_cv(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, float input_scale, int zero_point, int swapRB = 0)
{
	cv::Mat img;
	if (sample.channels() == 4)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
	}
	else if (sample.channels() == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
	}
	else if (sample.channels() == 3 && swapRB == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
		cout << "BGR2RGB!" << endl;
	}
	else
	{
		img = sample;
	}
	cv::Mat img2(img_h,img_w,CV_8UC3,cv::Scalar(0,0,0));
	int h,w;
	if((float)img.cols/(float)img_w > (float)img.rows/(float)img_h){
		h=1.0*img_w/img.cols*img.rows;
		w=img_w;
		cv::resize(img, img, cv::Size(w,h));
	}else{
		w=1.0*img_h/img.rows*img.cols;
		h=img_h;
		cv::resize(img, img, cv::Size(w,h));
	}
	int top = (img_h - h)/2;
	int bat = (img_h - h + 1)/2;
	int left = (img_w - w)/2;
	int right = (img_w - w + 1)/2;


	cv::copyMakeBorder(img,img2,top,bat,left,right,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
	uint8_t* img_data = img2.data;
	int hw = img_h * img_w;

	for (int h = 0; h < img_h; h++)
	{
		for (int w = 0; w < img_w; w++)
		{
			for (int c = 0; c < 3; c++)
			{
				int udata = *img_data;
				if (udata > 255)
					udata = 255;
				else if (udata < 0)
					udata = 0;
				input_data[c * hw + h * img_w + w] = udata;
				img_data++;
			}
		}
	}
}

void get_input_data_new(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, float input_scale, int zero_point, int swapRB = 0)
{
	cv::Mat img;
	if (sample.channels() == 4)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGRA2BGR);
	}
	else if (sample.channels() == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_GRAY2BGR);
	}
	else if (sample.channels() == 3 && swapRB == 1)
	{
		cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
		cout << "BGR2RGB!" << endl;
	}
	else
	{
		img = sample;
	}
	// cv::Mat img2(img_h,img_w,CV_8UC3,cv::Scalar(0,0,0));
	// int h,w;
	// if((float)img.cols/(float)img_w > (float)img.rows/(float)img_h){
	// 	h=1.0*img_w/img.cols*img.rows;
	// 	w=img_w;
	// 	cv::resize(img, img, cv::Size(w,h));
	// }else{
	// 	w=1.0*img_h/img.rows*img.cols;
	// 	h=img_h;
	// 	cv::resize(img, img, cv::Size(w,h));
	// }
	// int top = (img_h - h)/2;
	// int bat = (img_h - h + 1)/2;
	// int left = (img_w - w)/2;
	// int right = (img_w - w + 1)/2;


	// cv::copyMakeBorder(img,img2,top,bat,left,right,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
	uint8_t* img_data = img.data;
	int hwc = img_h * img_w * 3;
	for (int counter = 0; counter < hwc; counter++){
		input_data[counter] = *img_data;
		img_data++;
	}
	// for (int h = 0; h < img_h; h++)
	// {
	// 	for (int w = 0; w < img_w; w++)
	// 	{
	// 		for (int c = 0; c < 3; c++)
	// 		{
	// 			int udata = *img_data;
	// 			if (udata > 255)
	// 				udata = 255;
	// 			else if (udata < 0)
	// 				udata = 0;
	// 			input_data[c * hw + h * img_w + w] = udata;
	// 			img_data++;
	// 		}
	// 	}
	// }
}

layer make_darknet_layer(int batch, int w, int h, int net_w, int net_h, int n, int total, int classes)
{
	layer l = {0};
	l.n = n;
	l.total = total;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n * (classes + 4 + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.inputs = l.w * l.h * l.c;
	// yolov5
	float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

	l.anchors = ( float* )calloc(total * 2, sizeof(float));
		
	l.mask = ( int* )calloc(n, sizeof(int));
	if (9 == total)
	{
		for (int i = 0; i < total * 2; ++i)
		{
			l.anchors[i] = anchors[i];
		}
		if (l.w == net_w / 32)
		{
			int j = 6;
			for (int i = 0; i < l.n; ++i)
				l.mask[i] = j++;
		}
		if (l.w == net_w / 16)
		{
			int j = 3;
			for (int i = 0; i < l.n; ++i)
				l.mask[i] = j++;
		}
		if (l.w == net_w / 8)
		{
			int j = 0;
			for (int i = 0; i < l.n; ++i)
				l.mask[i] = j++;
		}
	}

	l.outputs = l.inputs;
	l.output = ( float* )calloc(batch * l.outputs, sizeof(float));

	return l;
}

// void free_darknet_layer(layer l)
// {
//     if (NULL != l.anchors)
//     {
//         free(l.anchors);
//         l.anchors = NULL;
//     }
//     if (NULL != l.mask)
//     {
//         free(l.mask);
//         l.mask = NULL;
//     }
//     if (NULL != l.output)
//     {
//         free(l.output);
//         l.output = NULL;
//     }
// }

static int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w * l.h);
	int loc = location % (l.w * l.h);
	return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
}

int yolo_num_detections(layer l, float thresh)
{
	int i, n, b;
	int count = 0;
	for (b = 0; b < l.batch; ++b)
	{
		for (i = 0; i < l.w * l.h; ++i)
		{
			for (n = 0; n < l.n; ++n)
			{
				int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
				if (l.output[obj_index] > thresh)
					++count;
			}
		}
	}
	return count;
}

int num_detections(vector<layer> layers_params, float thresh)
{
	int i;
	int s = 0;
	for (i = 0; i < ( int )layers_params.size(); ++i)
	{
		layer l = layers_params[i];       
		s += yolo_num_detections(l, thresh);

	}

	return s;
}

detection* make_network_boxes(vector<layer> layers_params, float thresh, int* num)
{
	layer l = layers_params[0];
	int i;
	int nboxes = num_detections(layers_params, thresh);
	if (num)
		*num = nboxes;
	detection* dets = ( detection* )calloc(nboxes, sizeof(detection));

	for (i = 0; i < nboxes; ++i)
	{
		dets[i].prob = ( float* )calloc(l.classes, sizeof(float));
	}
	return dets;
}

void correct_yolo_boxes(detection* dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if ((( float )netw / w) < (( float )neth / h))
	{
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else
	{
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i)
	{
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / (( float )new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / (( float )new_h / neth);
		b.w *= ( float )netw / new_w;
		b.h *= ( float )neth / new_h;
		if (!relative)
		{
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

box get_yolo_box(float* x, float* anchors, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;

	b.x = (i + (x[index + 0 * stride]) * 2 - 0.5) / lw;
	b.y = (j + (x[index + 1 * stride]) * 2 - 0.5) / lh;
	b.w = (4 * pow(x[index + 2 * stride], 2) * anchors[2 * n]) / w;
	b.h = (4 * pow(x[index + 3 * stride], 2) * anchors[2 * n + 1]) / h;

	return b;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int* map, int relative,
						detection* dets)
{
	int i, j, n, b;
	float* predictions = l.output;
	int count = 0;
	for (b = 0; b < l.batch; ++b)
	{
		for (i = 0; i < l.w * l.h; ++i)
		{
			int row = i / l.w;
			int col = i % l.w;
			for (n = 0; n < l.n; ++n)
			{
				int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
				float objectness = predictions[obj_index];
				if (objectness <= thresh)
					continue;
				int box_index = entry_index(l, b, n * l.w * l.h + i, 0);

				dets[count].bbox = get_yolo_box(predictions, l.anchors, l.mask[n], box_index, col, row, l.w, l.h, netw,
												neth, l.w * l.h);
				dets[count].objectness = objectness;
				dets[count].classes = l.classes;
				for (j = 0; j < l.classes; ++j)
				{
					int class_index = entry_index(l, b, n * l.w * l.h + i, 4 + 1 + j);
					float prob = objectness * predictions[class_index];
					dets[count].prob[j] = (prob > thresh) ? prob : 0;
				}
				++count;
			}
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}


void fill_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
						int* map, int relative, detection* dets)
{
	int j;
	for (j = 0; j < ( int )layers_params.size(); ++j)
	{
		layer l = layers_params[j];	
		int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
		dets += count;
		
	}
}

detection* get_network_boxes(vector<layer> layers_params, int img_w, int img_h, int net_w, int net_h, float thresh,
							int* map, int relative, int* num)
{
	// make network boxes
	detection* dets = make_network_boxes(layers_params, thresh, num);

	// fill network boxes
	fill_network_boxes(layers_params, img_w, img_h, net_w, net_h, thresh, map, relative, dets);
	return dets;
}

// release detection memory
void free_detections(detection* dets, int nboxes)
{
	int i;
	for (i = 0; i < nboxes; ++i)
	{
		free(dets[i].prob);
	}
	free(dets);
}

float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	float area = w * h;
	return area;
}

float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w * a.h + b.w * b.h - i;
	return u;
}

float box_iou(box a, box b)
{
	return box_intersection(a, b) / box_union(a, b);
}

void do_nms_sort(detection* dets, int total, int classes, float iou_thresh)
{
	int i, j, k;
	k = total - 1;
	for (i = 0; i <= k; ++i)
	{
		if (dets[i].objectness == 0)
		{
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k + 1;

	for (k = 0; k < classes; ++k)
	{
		for (i = 0; i < total; ++i)
		{
			dets[i].sort_class = k;
		}
		// qsort(dets, total, sizeof(detection), nms_comparator);
		for (i = 0; i < total; ++i)
		{
			if (dets[i].prob[k] == 0)
				continue;
			box a = dets[i].bbox;
			for (j = i + 1; j < total; ++j)
			{
				box b = dets[j].bbox;
				if (box_iou(a, b) > iou_thresh)
				{
					dets[j].prob[k] = 0;
				}
			}
		}  
	}
}

int check_file_exist(const char* file_name)
{
	FILE* fp = fopen(file_name, "r");
	if (!fp)
	{
		fprintf(stderr, "Input file not existed: %s\n", file_name);
		return 0;
	}
	fclose(fp);
	return 1;
}


static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

int set_graph(int net_h, int net_w, graph_t graph){
	int img_size = net_h * net_w * 3;
	int dims[] = {1, 3, net_h, net_w};    // nchw
 
	vector<uint8_t> input_data(img_size);

	tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
	if (input_tensor == nullptr)
	{
		fprintf(stderr, "Get input tensor failed\n");
		return -1;
	}

	if (set_tensor_shape(input_tensor, dims, 4) < 0)
	{
		fprintf(stderr, "Set input tensor shape failed\n");
		return -1;
	}

	if (prerun_graph(graph) < 0)
	{
		fprintf(stderr, "Prerun graph failed\n");
		return -1;
	}
	return 0;
}

int get_classes(graph_t graph){
	int dim[4];
	get_tensor_shape(get_graph_output_tensor(graph, 0, 0), dim, 4);
	int classes = dim[1]/3 - 5;
	return classes;
}
void run_graph_wrapper(int64_t p){
	std::cout << "Run graph wrapper!";
	printf("graph: %d", p);
	graph_t* graph = (graph_t*) p;
	run_graph(*graph, 1);
}

void postpress_graph_image_wrapper(int height, int width, float* array, graph_t graph, int output_node_num,
								   int net_w, int net_h, int classes, int num_dets, float nms, float thresh)
{
	// cv::Mat frame = cv::Mat(height, width, CV_8UC3, (uchar*)data_pointer);
	// const char *coco_names[80] = {"person","bicycle","car","motorbike","aeroplane","bus",
	// 							"train","truck","boat","traffic light","fire hydrant","stop sign",
	// 							"parking meter","bench","bird","cat","dog","horse","sheep","cow",
	// 							"elephant","bear","zebra","giraffe","backpack","umbrella","handbag",
	// 							"tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
	// 							"baseball bat","baseball glove","skateboard","surfboard","tennis racket",
	// 							"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
	// 							"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
	// 							"chair","sofa","pottedplant","bed","diningtable","toilet","tvmonitor",
	// 							"laptop","mouse","remote","keyboard","cell phone","microwave","oven",
	// 							"toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};
	std::vector<layer> layers_params;
	int numBBoxes = 3, total_numAnchors = 9;
	layers_params.clear(); 

	for (int i = 0; i < output_node_num; ++i)
	{
		tensor_t out_tensor = get_graph_output_tensor(graph, i, 0);    //"detection_out"
		int out_dim[4];
		get_tensor_shape(out_tensor, out_dim, 4);
		layer l_params;
		int out_w = out_dim[3];
		int out_h = out_dim[2];
		l_params = make_darknet_layer(1, out_w, out_h, net_w, net_h, numBBoxes, total_numAnchors,  classes);
		layers_params.push_back(l_params);

		/* dequant output data */
		float output_scale = 0.f;
		int output_zero_point = 0;
		get_tensor_quant_param(out_tensor, &output_scale, &output_zero_point, 1);
		int count = get_tensor_buffer_size(out_tensor) / sizeof(uint8_t);
		uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(out_tensor);
		float* data_fp32 = ( float* )malloc(sizeof(float) * count);
		for (int c = 0; c < count; c++)
		{
			data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale;
		}
		/* copy data to darknet layer */
		memcpy(( void* )l_params.output, ( void* )data_fp32, sizeof(float) * l_params.inputs * l_params.batch);

		free(data_fp32);
	}

	int nboxes = 0;
	// get network boxes
	detection* dets =
		get_network_boxes(layers_params, width, height, net_w, net_h, thresh, 0, relative, &nboxes);

	if (nms != 0)
	{
		do_nms_sort(dets, nboxes, classes, nms);
	}
	int counter = 0;
	int i, j;
	for (i = 0; i < nboxes; ++i)
	{
		int cls = -1;
		// char cvTest[64];
		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > thresh)
			{
				if (cls < 0)
				{
					cls = j;
				}
				// fprintf(stderr, "%d: %.0f%%\n", cls, dets[i].prob[j] * 100);
				array[counter*6 + 5] = (float) (dets[i].prob[j] * 100); // write confidence to detection;
				// sprintf(cvTest,"%s",coco_names[cls]);
			}

		}
		if (cls >= 0)
		{
			box b = dets[i].bbox;
			int left = (b.x - b.w / 2.)  * width;
			int top = (b.y - b.h / 2.) * height;
			// if (top < 30) {
			// 	top = 30;
			// 	left +=10;
			// }
 
			// fprintf(stderr, "left = %d,top = %d, width = %.1f, height = %.1f\n", left, top, b.w, b.h);
			array[counter*6] = (float) left;
			array[counter*6 + 1] = (float) top;
			array[counter*6 + 2] = (float) (b.w*width);
			array[counter*6 + 3] = (float) (b.h*height);
			array[counter*6 + 4] = (float) cls;
			counter++;
			// if (draw){
			// 	cv::Rect rect(left, top, b.w*frame.cols, b.h*frame.rows);

			// 	int baseline;
			// 	cv::Size text_size = cv::getTextSize(cvTest, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
			// 	cv::Rect rect1(left, top-20, text_size.width+10, 20);
			// 	cv::rectangle(frame,rect,obj_id_to_color(cls),2,20,0);
			// 	cv::rectangle(frame,rect1,obj_id_to_color(cls),-1);
			// 	cv::putText(frame,cvTest,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
			// }
		}

		if (dets[i].mask)
			free(dets[i].mask);
		if (dets[i].prob)
			free(dets[i].prob);
		if (counter == num_dets) {
			break;
		}
	}
	// set old detections to zero
	while (counter < num_dets){
		array[counter*6 + 0] = 0;
		array[counter*6 + 1] = 0;
		array[counter*6 + 2] = 0;
		array[counter*6 + 3] = 0;
		array[counter*6 + 4] = 0;
		array[counter*6 + 5] = 0;
		counter++;
	}

	free(dets);

	for (int i = 0; i < (int)layers_params.size(); i++)
	{
		layer l = layers_params[i];
		if (l.output)
			free(l.output);
		if (l.anchors)
			free(l.anchors);
		if (l.mask)
			free(l.mask);
	}
}

int set_image_wrapper(void* data_pointer, int height, int width, tensor_t input_tensor, int net_h, int net_w){
	float input_scale = 0.f;
	int input_zero_point = 0;
	int img_size = net_h * net_w * 3;
	vector<uint8_t> input_data(img_size);
	cv::Mat frame = cv::Mat(height, width, CV_8UC3, (uchar*)data_pointer);
	if (frame.empty())
	{
		fprintf(stderr, "Empty buffer\n");
		return -1;
	}
	get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
	if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
	{
		fprintf(stderr, "Set input tensor buffer failed\n");
		return -1;
	}
	get_input_data_new(frame,input_data.data(),net_w,net_h,input_scale, input_zero_point, 0);
	return 0;
}