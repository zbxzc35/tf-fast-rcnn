I've been working on transplanting Fast R-CNN to a TensorFlow version. And I came across 2 problems as below:

1. In [roi_pooling_layer.cpp](https://github.com/rbgirshick/caffe-fast-rcnn/blob/fast-rcnn/src/caffe/layers/roi_pooling_layer.cpp), the C++ implementation of RoI Pooling layer, a dynamic variable `num_rois` is used to control a loop to process each roi of each picture. Its value is stored in the original dataset and will only be valid during the graph is running. However, since the graph in TensorFlow has to be settled before it being run, I don't know what to put in the loop control as the iteration limit.

2. After the roi_pooling_layer, there follows a InnerProduct layer (in Caffe), the shape of whose input depends on the output shape of the roi_pooling_layer, which ultimately depends on `num_rois`. So the shape of the corresponding `weight_variable` and `bias_variable` for the following operation `matmul`, which is going to act as the InnerProduct layer, cannot be settled before the graph is running.

In addition, in the RoI Pooling layer, there are some other loop control variables based on the former calculation, like `hstart`, `hend`, `wstart` and `wend`. They are even harder to get valid values before the graph is built, because the `num_rois` is part of the input after all.

So may I ask if TensorFlow supports Fast R-CNN (currently)? Or are there any possible specific tips to solve these problems?

Thank you!
