# action-recognition

#### Usage

Extracts frames/flow from a root directory containing videos/videofolders in its sub-directories. 
Outputs will be saved in matching pathway, but with *root* replaced with *out*

1. Put dataset in *Data* directory

2. Extract frames 
   ```
   python videoprocessing/extract_frames.py --root ucf/ucf_vids --out ucf/ucf_imgs --vid_ext .mp4 --imgs_ext .jpg -r 30
   ```
3. Extract flow (optional) 
   ```
   python videoprocessing/extract_flow.py --root data/ucf/ucf101_imgs --out data/ucf/ucf101_flow --method tvl1 --cuda --workers 1 
   ```
4. Build labels 
   ```
   python arcore/get_labels.py --root_imgs data/data_ucf/ucf_imgs --root_flow data/data_ucf/ucf_flow --ds ucf --out data/data_ucf --test_list_path ucf/ucfTrainTestlist/trainlist02.txt
   ```