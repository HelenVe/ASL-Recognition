# ASL-Recognition
**American Sign Language Recognition Thesis Using RNN + CTC Loss**

This repository contains my thesis for American Sign Language Recognition.
We have 2 models: a static model for frame-level letter recognition and a sequential model complemented with CTC loss to recognize variable length  sequences of letters

**Dataset**: 
Chicago FS Vid  (ttic.edu/livescu/chicago-fsvid.tgz) -> ASRU 2017

The dataset contains 4 users, each has 2 Matlab Files.
From these files we extract and save the hand images as well as the HoG Features, and create csv files with the path and the corresponding labels.
The labels have 2 formats: 
- One letter for each frame
- One word for a sequence of frames

**How to run**:
1. Inside your main project folder create a "data and a "hog" folder. Inside these two create 4 folders for the users: "andy", "drucie", "rita", "robin".
2. Inside your main project folder create a csv folder to save all the csv files.
3. The get_data.py script extracts the Images from the corresponding Matlab files so run
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
 
 **Optical Flow Calculation**
 
 You can calculate the optical flow of the images and save to a folder.
 1. Manually create "optical_flow" folder inside main project folder. Inside this, create 4 folders for the users.
 2.  Run 
  ```
  python3 optical_flow.py -i data/rita -s optical_flow/rita
  ```
  This code calculates the optical flow of the images using OpenCV's Gunnar Farneback Dense optical flow inside "rita" and saves them.

 **Feature Extraction**
 
 For the feature extraction part we use Pytorch Resnet18 Pretrained on Imagenet.
 We can extract features from the initial and the optical flow images.
 1. Inside the main project folder create a folder named "resnet" and folder "optical_flow_features" or whatever you want to save the features. Again, create 4 folders for the users.
 To extract features from the initial images:
 ```
 python3 get_features.py -i data/rita -s resnet
 ```
 To extract features from the optical flow of the images:
 ```
  python3 get_features.py -i optical_flow/rita -s optical_flow_features
 ```
 Before training remember to split the data for each user into train, dev, test so the code can work, else you must make some changes.
 You can also create a folder for all the users, add their images and split into train, dev, test.
 
 **Static Model**
 
 To train the static model run 
 ```
 python3 frame_level_train.py -u [andy, drucie, rita, robin]
 ```
 and input "1". To test it input "2" when prompted. Some images and their predicted values wlll be displayed.
 You can always change the layers. The data is loaded from the data_generator according to the batch size.
 The fsvid.csv loads the labels as well. 
 Inside the main project folder create a "models" folder. Create subfolders. In lines 129, 133, 145 change the path name accordingly in order ot save the models.
 
 **Sequential Model**
 
 **Training**
 
 To train the Resnet based sequential model run 
 ```
 python3 train_resnet.py -u [andy, drucie, rita, robin]
 ```
This model takes the 512-dim feature vectors of each frame output by Resnet and trains a model.

To train the HoG based sequential model run 
 ```
 python3 train_hog.py -u [andy, drucie, rita, robin]
```
This model takes the 128-dim HoG feature vectors of each frame and trains a model.
Keep in mind that if you changed any of the folder names above, you should check the code because these values have been hardcoded.
Will fix in the future.

**Testing**

To test the sequential models, change the code to load the model you want from the "models" folder. The metric used is the Word Error Rate.
We use Beam Search Decoding with beam width = 5 to decode.
Run 
```
python3 test.py 
```
**DEMO**

There is also a live demo with the end-to-end VGG model to recognise the letters. Run
```
python3 live_demo.py
```


