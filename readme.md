
Date: Jan 10th 18

Folders HAS BEEN COMBINED

MAJOR FILES CAN BE FOUND HERE

This directory contains 6 parts--Final_Submission, Presentation_final, Proposal, IMG_2457.MOV, handwritten_recogination.pdf, and readme.md

- Final_Submission: this folder contains all documents for final submission. This folder has five major parts.

- - conv_tensorflow: this folder contains codes for the Convolutional Neural Network in Tensorflow. File tensorflow_train_test_model.py is the training code of the network. To run this .py file, type "python tensorflow_train_test_model.py". Weights and biases will be saved automatically in weight_save folder. File tensorflow_test_model.py is the testing code of the network by loading saved weights and biases from weight_save_XXXX folder. To run this .py file, type "python tensorflow_test_model.py" This .py file will also generate the C include files for weights and biases (will be included in the DSP C model). This folder also contains training and testing data and labels, as well as file of display-characters on monitor. There are three weight_save folders. The weight_save_final was used for final demo. The weight_save_128_work was used and worked on DSP. The weight_save_512 was unable to be loaded to DSP because of over-loading problem.

- - data_generation: this folder contains the code, dataset.py, for generating training and testing data and labels from the Chars74k dataset and my own data collection. It also generates the txt file for display-characters on monitor. To run this .py file, type "python dataset.py". This code involved many different sections. To use it, you need to make small modifications on threshold percentages, and comment/uncomment some sections of code. If you have any question regarding using this .py file, please contact me at yiyuezha@usc.edu. The other three folders are images of display-characters, The Chars74k Dataset, and the collected data.

- - final_project.pdf: this is the final report of my project in pdf form. 

- - final_report: this folder contains all source documents (Latex and figures) to generate the final_project.pdf

- - main.c: It contains all the required files for it to work on DSP. The main.c consists the C model of the Convolutional Neural Network. The Debug folder contains all the include files (weights and biases), generated from the tensorflow_test_model.py in folder conv_tensorflow. To use it, please load the folder to workspace and follow usage in the final_project.pdf


- Presentation_final: this folder contains the final presentation powerpoint in pdf form

- Proposal: this folder contains the proposal of the project and all the source documents

- IMG_2457.MOV: this is the video recording of the demo for the project. DUE TO FILE TOO LARGE, CONTACT ME IF YOU WANT TO SEE IT

- handwritten_recogination.pdf: this is the proposal presentation powerpoint in pdf form
