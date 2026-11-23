Automatic License Plate Detection
This repository contains our CSE281 Digital Image Processing project. We
are working on a system that finds the license plate in a vehicle image
using traditional image-processing methods.

The goal is to locate the plate and mark it with a bounding box. We are
not trying to read the letters or numbers on the plate.
Project idea
Finding a license plate in an image can be difficult when the photo has
poor lighting, a different camera angle, a busy background, or noise. In
this project, we will test whether classical image-processing techniques
can locate the plate under these conditions.

For each image, the system is expected to return: - A copy of the image
with a box around the detected plate. - The box coordinates (x, y,
width, and height). - The area of the box in pixels.

We will use the predicted box and the dataset's annotation to measure
how close the detection is to the actual plate.
Dataset
We plan to use the Car License Plate Detection Dataset mentioned in
the CSE281 project specification. The specification describes it as
having around 433 vehicle images and 471 license-plate bounding-box
annotations.

Before using it for the experiments, we still need to check a few
things: - The downloaded images open correctly. - The annotation files
match the correct images. - Every image in our test set has a
ground-truth bounding box. - The dataset licence allows academic use. -
The images include enough variation, such as changes in lighting, angle,
and background.

Dataset download/source: To be added after the team confirms the
exact dataset source.

Licence: We will record the licence here after checking the original
dataset page.

We will not treat the dataset as fully verified until these checks are
complete.
Planned method: Pipeline B
We chose Pipeline B as the approach we plan to implement. It uses a
sequence of image-processing steps to find possible plate regions and
then remove candidates that do not have suitable dimensions or shape.
1. Load the image -- Read the vehicle image.
2. Convert to grayscale -- Work with intensity values instead of
   the original colour channels.
3. Reduce noise -- Apply a Gaussian or median filter.
4. Find edges -- Use the Sobel operator to highlight strong changes
   in intensity.
5. Apply thresholding -- Create candidate regions from the
   processed image.
6. Find connected regions -- Identify groups of connected pixels
   that may contain a plate.
7. Filter the candidates -- Compare their area, width, height, and
   aspect ratio to remove unlikely regions.
8. Draw the result -- Mark the selected plate region with a
   bounding box and report its coordinates and area.
The filter sizes and thresholds will be tested and adjusted using the
development data. We will document the settings that work best and the
cases where the method fails.
How we will evaluate the result
Our main evaluation metric will be Intersection over Union (IoU). It
compares the predicted bounding box with the ground-truth box provided
by the dataset.
\[
`\frac{\text{Area of overlap}}{\text{Area covered by both boxes}}`{=tex}
\]
An IoU of 1 means the boxes match perfectly, while an IoU of 0 means
they do not overlap.

For this project, we plan to count a detection as successful when its
IoU is at least 0.50. This is the threshold we have selected for our
evaluation; it is not a threshold imposed by the course.

We will also record: - Detection rate: the percentage of test images
where the system finds the plate successfully. - False detections:
cases where the system marks a region that is not a license plate. -
Processing time: how long the system takes to process the images.
Tools
We expect to use: - Python - NumPy - OpenCV - scikit-image - SciPy -
Matplotlib - Jupyter Notebook or Google Colab

The project will use classical image processing. We will not use
deep-learning models, neural networks, pretrained object detectors, or
OCR.
Team
No. Name          Responsibility

  1 You           Tech Lead
  2 Jooks (AIU)   Data and Baseline Engineer
  3 Haneen        Enhancement Engineer
  4 Judy          Restoration and Frequency Engineer
  5 Malak Emara   Segmentation Engineer
  6 Habiba        Morphology and Evaluation Engineer
Repository layout
Automatic-License-Plate-Detection/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── 01_dataset_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── enhancement.py
│   ├── segmentation.py
│   ├── detection.py
│   ├── evaluation.py
│   └── pipeline.py
├── experiments/
├── results/
└── report/
We may change this structure as the project develops. We will avoid
uploading the full dataset unless its licence and file size allow us to
do so. If the data stays outside GitHub, the data/README.md file will
explain how to obtain it.
Setup
Install the libraries with:
pip install numpy opencv-python scikit-image scipy matplotlib jupyter
Project status
This is the initial project plan. The dataset and its annotations still
need to be checked, and the detection pipeline has not yet been
implemented. We will add the code, experiments, and results as we work
through the project.
Project constraints
- Use classical image-processing methods.
- Do not use OCR or character recognition.
- Do not use neural networks, deep learning, or pretrained object
  detectors.
- Make sure every test image has a valid ground-truth bounding box.
- Keep the complete test-set processing time below 10 minutes on the
  free Google Colab CPU environment, as required by the course project
  specification.
