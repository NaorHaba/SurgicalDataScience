# SurgicalDataScience

This repository was created as a final project to the Technion course _Advanced Topics in Data Science (Surgical Data Science)_. <BR>
In this project we created models to the task of Gesture Recognition in Surgical Procedures. 
Parts of this code were provided by the course's staff and were adapted by us, to enhance and improve models and functionality.

To run the offline feature extraction of the frames from all surgery videos, you need to run `feature_extraction.py`

To run the model on all the splits you need to run the `train_experiment.py` file (make sure test_split parameter is set to None for all splits).

You can change any of the parameters in the args object to control the architecture and training parameters.

Good Luck :)