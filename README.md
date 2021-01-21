README.md

These scripts and data provide example code for the method described in:

Haskell et al. MRM 2019, "Network Accelerated Motion 
Estimation and Reduction (NAMER): Convolutional neural network guided 
retrospective motion correction using a separable motion model"

To start, run "train_namer_cnn.py". This script requires that keras with
TensorFlow backend is installed on your machine. Next, move the final 
trainded model from the "models" folder to the main directory, and update 
the variable "cnn_model_name" in "namer_recon.m" with the proper name. 
Third, run "namer_recon.m" to run the NAMER method in MATLAB.

Here a previously trained network "namer_trained_model.h5" is included if 
you want to skip the CNN training step.

The empty directories "convergence_curves", "model_weights", and "models"
start as empty but are used by the "train_namer_cnn.py" script.

This example was most recently tested using matlab 2017b, and uses that 
versions's syntax for optimization settings, etc.


Key scripts

train_namer_cnn.py- This script constructs and trains the CNN, using the 
                    training data in "training_data.mat". When it completes
                    training, move the model from the "models" folder to 
                    the main directory where the "namer_recon.m" script is.

namer_recon.m - This script performs the separable cost function 
                version of the NAMER method (Eqn 3 in Haskell et al. 2019),
                and corresponds to the result shown in the bottom left of 
                Figure 4-B in the paper.

run_namer_cnn.py -  This script evaluates all of the patches for a given
                    input image and returns the output of the motion 
                    artifact detecting CNN.







