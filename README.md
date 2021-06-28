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

**How to run**:
1. Inside your main project folder create a "data and a "hog" folder. Inside these two create 4 folders for the users: "andy", "drucie", "rita", "robin".
2. The get_data.py script extracts the Images from the corresponding Matlab files so run
  ```
  python3 get_data.py -u [andy, drucie, rita, robin]
  ```
  For example for Andy run python3 get_data.py -u andy 
  
3. The get_hog.py scripts extracts the hog features and places them inside the hog folder. Run
```
python3 get_hog.py -u [andy, drucie, rita, robin].
```
 Because the data is split into 10 folds, the script creates 10 folds inside the users folder under the name fold_num. Each fold contains 54 words.
 Then the script creates "new_fold_num"  with the words seperated into folders by moving the contents of the fold_num. Each file is save in .npy format.
 
 You must then manually seperate the data the way you want to and delete the empty folders fold_0 - fold_9
 
 ** Optical Flow Calculation **
 
 You can calculate the optical flow of the images and save to a folder.
 1. Manually create "optical_flow" folder inside main project folder. Inside this, create 4 folders for the users.
 2.  Run 
  ```
  python3 optical_flow.py -i data/rita -s optical_flow/rita
  ```
  This code calculates the optical flow of the images using OpenCV's Gunnar Farneback Dense optical flow inside "rita" and saves them.
