# Zero Shot Gesture Recognition (IE 590) Project

Please refer to [ZeroShot_Gesture_Recognition_Report](https://github.com/kmanchel/ZeroShot_Gesture_Recognition/blob/master/Report_Zero_Shot.pdf) for a detailed walk through of this project.

Please also note that the video dataset used for this project is available by contacting professor [Jun Wan](http://www.cbsr.ia.ac.cn/users/jwan/database/isogd.html). The associated annotations for this dataset are found [here](https://github.com/kmanchel/ZeroShot_Gesture_Recognition/blob/master/annotated_videos.csv).

Tensorflow version: v2.0.0

# 0. Layout

```bash
.
├── README.md
├── data
│   └── store_sample_data_here.png
├── ie590_project
│   ├── __pycache__
│   │   ├── build_dataset.cpython-36.pyc
│   │   ├── fit_and_predict.cpython-36.pyc
│   │   ├── fit_and_predict_notebook.cpython-36.pyc
│   │   ├── input_fn.cpython-36.pyc
│   │   ├── input_fn.cpython-37.pyc
│   │   ├── model_fn.cpython-36.pyc
│   │   └── predict.cpython-36.pyc
│   ├── experiments
│   │   └── ex1
│   │       └── params.json
│   ├── fit_and_predict.py
│   ├── fit_and_predict_notebook.py
│   ├── model
│   │   ├── both_hands_model_fc.data-00000-of-00002
│   │   ├── both_hands_model_fc.data-00001-of-00002
│   │   ├── both_hands_model_fc.index
│   │   ├── both_hands_model_lstm.data-00000-of-00002
│   │   ├── both_hands_model_lstm.data-00001-of-00002
│   │   ├── both_hands_model_lstm.index
│   │   ├── checkpoint
│   │   ├── input_fn.py
│   │   ├── m_iterative_model_fc.data-00000-of-00002
│   │   ├── m_iterative_model_fc.data-00001-of-00002
│   │   └── model_fn.py
│   ├── predict.py
│   ├── setup
│   │   ├── gen_bbox.py
│   │   └── v2f.py
│   ├── train.py
│   └── utils
│       ├── __pycache__
│       │   ├── utils.cpython-36.pyc
│       │   ├── utils.cpython-37.pyc
│       │   ├── utils_notebook.cpython-36.pyc
│       │   └── utils_notebook.cpython-37.pyc
│       ├── build_features.py
│       ├── feature_params.json
│       ├── utils.py
│       └── utils_notebook.py
├── Descriptor_Training
│   ├── Both_Hands\ 2.ipynb
│   ├── Both_Hands.ipynb
│   ├── F_Index\ 2.ipynb
│   ├── F_Index.ipynb
│   ├── F_Middle.ipynb
│   ├── F_Pinky\ 2.ipynb
│   ├── F_Pinky.ipynb
│   ├── F_Ring\ 2.ipynb
│   ├── F_Ring.ipynb
│   ├── F_Thumb\ 2.ipynb
│   ├── F_Thumb.ipynb
│   ├── M_Back\ 2.ipynb
│   ├── M_Back.ipynb
│   ├── M_Down\ 2.ipynb
│   ├── M_Down.ipynb
│   ├── M_Front.ipynb
│   ├── M_In\ 2.ipynb
│   ├── M_In.ipynb
│   ├── M_Iterative\ 2.ipynb
│   ├── M_Iterative.ipynb
│   ├── M_Out\ 2.ipynb
│   ├── M_Out.ipynb
│   ├── O_Back\ 2.ipynb
│   ├── O_Back.ipynb
│   ├── O_Down\ 2.ipynb
│   ├── O_Down.ipynb
│   ├── O_Front\ 2.ipynb
│   ├── O_Front.ipynb
│   ├── O_In\ 2.ipynb
│   ├── O_In.ipynb
│   ├── O_Out\ 2.ipynb
│   ├── O_Out.ipynb
│   ├── O_Up\ 2.ipynb
│   └── O_Up.ipynb
├── EDA.ipynb
├── KNN.ipynb
├── NEW_PIPELINE_DEMO.ipynb
├── ResNet_Feature_Map.ipynb
├── annotated_videos.csv
├── annotation_pipeline
│   ├── Annotations_New.ipynb
│   ├── Append_Descriptors.ipynb
│   ├── annotated_kaushik_done5.csv
│   ├── annotated_kaushik_mean.json
│   ├── autoreload.ipynb
│   ├── play_video_on_nb.ipynb
│   └── top50_videos_w_descriptors.csv
├── getting_unseen.ipynb
├── input\ pipeline\ testing.ipynb
└── old_csvs
│       ├── data_descriptors_mean.csv
│       ├── data_descriptors_mode.csv
│       ├── oversampled_m_down.csv
│       ├── predictions_mode.csv
│       ├── train.csv
│       ├── unseen_annotations.csv
│       ├── unseen_predictions.csv
│       └── validation.csv
└── requirements.txt

```

# 1. Setup

## 1.1 Build a virtual environment

1. install virtual 
```bash
pip install virtualenv
```

2. install python virtual environment named 'ie590_project_venv'
```bash
python3 -m venv ie590_project_venv
```

3. activate the virtual environment module
```bash
source ie590_project/bin/activate
```

4. when finish working on the venv
```bash
deactivate
```


# 2. When making a pull request to Github

```diff
- Note: Use a branch named by your github account. Do not push to the master branch.
```

1. setup git on the working directory (you only do it once)
```bash
cd posture_venv # your virtual venv directory
mkdir posture 
cd posture # make sure you are in the directory which will be your working directory

git init
git remote add origin https://github.com/msc11/ie590_project.git
git pull origin master
```

2. create a branch (you only do it once)
```bash
git branch <your_github_account_name>
git checkout <your_github_account_name>
```

3. update codes

4. push your update to your branch in github

```bash
git add <updated_file_names>
git commit -m "<message>"
git push origin <your_github_account_name>
```

5. make a pull request in the Github repository website

# 3. Feature Extraction Using Pretrained VGG16

## 1. Navigate to "../ie590_project/ie590_project/utils" in terminal
## 2. Modify features_params.json
### 1. Change "csv_path" to point to "../../jupyter_notebook/annotated_videos.csv"
annotated_videos.csv is a csv that contains the following information for each training/testing example: RGB Video Directory Path (M), Depth Video Directory Path (K), Gesture Label, and 19 Descriptor Annotations.
### 2. Change "dataset_path" to point to the directory where all the gesture videos are stored. eg "../ie590_project/data/IsoGD_phase_1
The directory structure must be identical to what was provided in the ChaLearnIsoGD2016 zip files. 
### 3. Change "features_path" to tell the script where to create feature files for each video. eg "../ie590_project/data/feature_maps/" (this directory has to be created beforehand)
The script will output a numpy file (.npy) for the features extracted from each and every video part of the "annotated_videos.csv"
## 3. Run the feature extraction script
```bash
python3 build_features.py --USE_GPU=True
```
The script takes the **--USE_GPU==True** argument if you want to use GPU to extract features. It is set to **False** by default. 
Once extracted, feature files will be written into the provided "features_path". 
Also note that the annotated_videos.csv file will be updated with the feature file paths for each video. 



