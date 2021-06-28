# ASL-Recognition
**American Sign Language Recognition Thesis Using RNN + CTC Loss**

This repository contains my thesis for American Sign Language Recognition.

**Dataset**: 
Chicago FS Vid  (ttic.edu/livescu/chicago-fsvid.tgz) -> ASRU 2017

The dataset contains 4 users, each has 2 Matlab Files.
From these files we extract and save the hand images as well as the HoG Features, and create csv files with the path and the corresponding labels.
The labels have 2 formats: 
- One letter for each frame
- One word for a sequence of frames
