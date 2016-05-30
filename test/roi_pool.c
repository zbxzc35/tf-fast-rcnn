#include <math.h>

int imax(int a, int b)
{
	return (a > b) ? a : b;
}

int imin(int a, int b)
{
	return (a < b) ? a : b;
}

typedef struct
{
	float data[256][36];
}rpc;

rpc roi_pool(float *feature_maps, float *input_roi)
{
	int i = 0;
	int j = 0;
	int c = 0;
	int ph = 0;
	int pw = 0;
	int h = 0;
	int w = 0;
	
	rpc roi_pool_conv5;
	
	for (i = 0; i < 256; i++)
	{
		for (j = 0; j < 36; j++)
		{
			roi_pool_conv5.data[i][j] = 0;
		}
	}
	
	float pooled_height = 6;
	float pooled_width = 6;
	float spatial_scale = 0.0625;
	
	int height = 54;
	int width = 38;
	int channels = 256;
	
	float roi_start_w = round(input_roi[1] * spatial_scale);
	float roi_start_h = round(input_roi[2] * spatial_scale);
	float roi_end_w = round(input_roi[3] * spatial_scale);
	float roi_end_h = round(input_roi[4] * spatial_scale);
	
	float roi_height = imax(roi_end_h - roi_start_h + 1, 1);
	float roi_width = imax(roi_end_w - roi_start_w + 1, 1);
	float bin_size_h = roi_height / pooled_height;
	float bin_size_w = roi_width / pooled_width;
	
	
	for (c = 0; c < channels; c++)
	{
		for (ph = 0; ph < pooled_height; ph++)
		{
			for (pw = 0; pw < pooled_width; pw++)
			{
				int hstart = floor(ph * bin_size_h);
				int wstart = floor(pw * bin_size_w);
				int hend = ceil((ph + 1) * bin_size_h);
				int wend = ceil((pw + 1) * bin_size_w);
				
				hstart = imin(imax(hstart + roi_start_h, 0), height);
				hend = imin(imax(hend + roi_start_h, 0), height);
				wstart = imin(imax(wstart + roi_start_w, 0), width);
				wend = imin(imax(wend + roi_start_w, 0), width);
				
				int pool_index = ph * pooled_width + pw;
				
				for (h = hstart; h < hend; h++)
				{
					for (w = wstart; w < wend; w++)
					{
						int fm_index = (h * height + w) * width + c;
						if (*(feature_maps + fm_index) > roi_pool_conv5.data[c][pool_index])
						{
							roi_pool_conv5.data[c][pool_index] = *(feature_maps + fm_index);
						}
					}
				}
			}
		}
	}
	
	return roi_pool_conv5;
}
