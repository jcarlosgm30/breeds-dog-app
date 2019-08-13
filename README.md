# Project: dog-recognition-app

In this project I have developing an algorithm that could be used as part of a mobile or web app. The app accepts any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling.

This project is from [Udacity Machine Learning Engineer Nanodegre](https://eu.udacity.com/).

For more info see `breeds-dog-app.ipynb`

### Install

This project requires **Python 2.7** and the following libraries installed:

- opencv-python==3.2.0.6
- h5py==2.6.0
- matplotlib==2.0.0
- numpy==1.12.0
- scipy==0.18.1
- tqdm==4.11.2
- keras==2.0.2
- scikit-learn==0.18.1
- pillow==4.0.0
- ipykernel==4.6.1
- tensorflow==1.0.0
- tensorflow-gpu==1.0.0

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Files

- `breeds-dog-app.ipynb`: notebook file with coding and explanation about the project. 

- `extract_bottlenetck_features.py` Python file to extract bottleneck features with differents pre-trained neural network.

- `haarcascades/haarcascade_frontalface_alt.xml` OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.

### Run

In a terminal or command window, navigate to the top-level project directory `breeds-dog-app/` (that contains this README) and run one of the following commands:

```bash
ipython notebook breeds-dog-app.ipynb
```  
or
```bash
jupyter notebook breeds-dog-app.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.
