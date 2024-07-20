# Notes

I tried implementing the following:
Input Image -> [Convolutional Layer] -> [Activation Layer] -> [Pooling Layer] -> [Fully Connected or Dense Layer] -> [Output]

Ended up with the following:
Input Image -> [Convolutional Layer] -> [Activation Layer] -> [Pooling Layer] -> [Convolutional Layer] -> [Activation Layer] -> [Pooling Layer] -> [Convolutional Layer] -> [Activation Layer] -> [Fully Connected or Dense Layer] -> [Output]

## Observations

After trying different combinations of layers, it seems adding 3 convulution layers, 3 activation layers, and 2 pooling layers gets us closest to 100% with an efficient 6ms/step. The model was more accurate with no pooling layers but it was too inefficient

The pooling layer I used is called Max Pooling. I did try Average Pooling but found Max Pooling to give the best accuraccy

For the last layer before output, I used a Dense Layer. Although I had to flatten the input first before I could pass it to the Dense Layer