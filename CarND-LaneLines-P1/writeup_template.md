# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
I classified the line segments from hough transformation into left lane marking and right lane marking, and extract the median of corresponding slope and intersections. at last only draw the line with median slope and intersection



### 2. Identify potential shortcomings with your current pipeline
there are still dirty line segments which makes the final detected median not stable. with a better parameter tunning, it will look better


### 3. Suggest possible improvements to your pipeline
maybe integrating the possible correlation between the frames(lanes in different frames) can improve the results
