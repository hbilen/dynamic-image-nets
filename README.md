# Dynamic Image Networks for Action Recognition
## Improved Results (see the extended version of CVPR paper)


ResNeXt-50        | HMDB51 (%) | UCF101 (%) |
------------------|--------|--------|
SI                |  53.5  |  87.6  |
DI                |  57.3  |  86.6  |
OF                |  55.8  |  84.9  |
DOF               |  58.9  |  86.6  |
SI+OF             |  67.5  |  93.9  |
SI+DI             |  61.3  |  90.6  |
OF+DOF            |  62.6  |  89.1  |
SI+DI+OF+DOF      |  71.5  |  95.0  |
SI+DI+OF+DOF+iDT  |  74.2  |  95.4  |

* Results are in the standard average multi-class accuracy (%)
* SI: RGB image
* DI: dynamic RBG image
* OF: optical flow 
* DOF: dynamic optical flow 
* iDT: improved trajectory features 


## Installation
1. Clone the Dynamic Image Net repository:

    ```Shell
    git clone --recursive  https://github.com/hbilen/dynamic-image-nets
    ```
    
2. Compile matconvnet toolbox: (see [http://www.vlfeat.org/matconvnet/install/](http://www.vlfeat.org/matconvnet/install/))

3. Download your dataset : (e.g. UCF101 from [http://crcv.ucf.edu/data/UCF101.php](http://crcv.ucf.edu/data/UCF101.php))

4. Convert videos to frames, resize them to 256x256 and store them in such a directory structure:
    
    ```Shell
    data/UCF101/ucfTrainTestlist/
    ├── classInd.txt
    ├── testlist01.txt
    ├── testlist02.txt
    ├── testlist03.txt
    ├── trainlist01.txt
    ├── trainlist02.txt
    └── trainlist03.txt
    data/UCF101/frames/
    ├── v_ApplyEyeMakeup_g01_c01
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   ├── 00003.jpg
    │   ├── 00004.jpg
    │   ├── 00005.jpg
    ```

## Compute and Visualise Approximate Dynamic Images
1. If you want to compute approximate dynamic images, get a list of ordered frames from a video and try
  ```matlab
  di = compute_approximate_dynamic_images(images) ;
  ```

2. If you want to visualise approximate dynamic images, get a list of ordered frames from a video and try
  ```matlab
  visualize_approximate_dynamic_images(images)
  ```

## Train 
1. Write your own `cnn_dataset_setup_data` or `cnn_ucf101_setup_data` to build your database (imdb):
2. Now you can train your model by running 

    ```matlab
    [net, info] = cnn_dicnn(opts)
    ```
## Evaluation

1. Download a trained model from the following link:
https://drive.google.com/open?id=0B0evBVYO74MEa29kZDQ2UlNDS1k

2. Set the appropriate opts parameters (e.g. opts.modelPath)

3. Run info = cnn_dicnn_evaluate(opts)


## Citing Dynamic Image Networks

If you find the code useful, please cite:

        @inproceedings{Bilen2016a,
          author    = "Bilen, H. and Fernando, B. and Gavves, E. and Vedaldi, A. and Gould, S.",
          title     = "Dynamic Image Networks for Action Recognition",
          booktitle = "CVPR",
          year      = "2016"
        }
        @article{Bilen2016c,
          author    = "Bilen, H. and Fernando, B. and Gavves, E. and Vedaldi, A.",
          title     = "Action Recognition with Dynamic Image Networks",
          journal   = "arXiv",
          year      = "2016"
        }
## License
The analysis work performed with the program(s) must be non-proprietary work. Licensee and its contract users must be or be affiliated with an academic facility. Licensee may additionally permit individuals who are students at such academic facility to access and use the program(s). Such students will be considered contract users of licensee. The program(s) may not be used for commercial competitive analysis (such as benchmarking) or for any commercial activity, including consulting.

