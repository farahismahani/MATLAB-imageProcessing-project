MATLAB Image Processing & Deep Learning Project
A MATLAB project covering image processing, object detection, and deep learning-based image enhancement. Developed as part of a group assignment.

Project Overview  
This project explores various image processing techniques, including histogram equalization, edge detection, and object detection using Hough Transform. It also integrates deep learning using a Context Aggregation Network (CAN) for image noise removal.  

The project is divided into three main parts:  
- Part A: Basic Image Processing (filters, histograms, edge detection)  
- Part B: Object Detection (circle detection using Hough Transform)  
- Part C: Deep Learning for Image Enhancement (CAN network for noise removal)  

An interactive MATLAB GUI is included for testing different image processing functions.  

Features  
i.   Load & Process Grayscale Images – Image display, filtering, contrast adjustment  
ii.  Edge Detection – Sobel, Canny, Prewitt, Roberts gradient operators  
iii. Histogram Equalization & Thresholding – Enhancing image contrast  
iv.  Circle Detection – Using Hough Transform to identify circular objects  
v.   Deep Learning for Image Noise Removal – Applying CAN (Context Aggregation Network)  
vi.  Interactive MATLAB GUI – Test image processing techniques in a user-friendly interface  

How to Run the Project  
 A. Clone the Repository  
```sh
git clone https://github.com/yourusername/MATLAB-ImageProcessing-Project.git
cd MATLAB-ImageProcessing-Project
```
 B. Open MATLAB & Run Scripts  
- Part A (Basic Processing):  
  ```matlab
  run('PartA_BasicProcessing/main.m')
  ```
- Part B (Circle Detection):  
  ```matlab
  run('PartB_CircleDetection/detectCircles.m')
  ```
- Part C (Deep Learning for Noise Removal):  
  ```matlab
  run('PartC_DeepLearning/runCAN.m')
  ```
- To test with different images, place them in the `TestImages/` folder.  

Requirements  
✅ MATLAB R2021b or newer  
✅ Image Processing Toolbox  
✅ Deep Learning Toolbox  

Screenshots / GIFs  
🖼️ GUI Interface PART A <img width="362" alt="gui" src="https://github.com/user-attachments/assets/67110d04-cb60-44fd-8594-571685419a63" />

🖼️ GUI Interface PART B ![image](https://github.com/user-attachments/assets/345923dd-e8e7-4708-be03-1d18e074e07c)

🖼️ Output of Trained Multiscale CAN  ![image](https://github.com/user-attachments/assets/6047e325-e0a1-440c-8cc8-6ac44d44ef13)

Developed by:  
- farah ismahani
- nisa syarafana
- abdul mujar
- nurul anis
- phrince powlgreat

Special thanks to UNIMAS, Associate Professor Dr. Teh Chee Siong, and MATLAB Documentation.  
