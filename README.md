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

3. Install additional matconvnet packages
    
  ```Shell
    run matconvnet/matlab/vl_setupnn.m ;
    vl_contrib install mcnExtraLayers ; vl_contrib setup mcnExtraLayers ;
    vl_contrib install autonn ; vl_contrib setup autonn ;
  ```

4. Download your dataset : (e.g. UCF101 from [http://crcv.ucf.edu/data/UCF101.php](http://crcv.ucf.edu/data/UCF101.php))

5. Convert videos to frames, resize them to 256x256 and store them in such a directory structure:
Alternatively, you can download RGB and precomputed optical flow frames from [Christoph Feichtenhofer](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/) and copy RGB frames under "UCF101/frames" and optical flow frames under "UCF101/tvl1_flow".
    
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

## Train a Dynamic Image Net
You can modify the options in `main_train.m` and train your model by running
    ```matlab
    main_train
    ```
    
Note: If you want to train a model on a different dataset than UCF101 or HMDB51, you need to write a custom script `cnn_dataset_setup_data` to build your database (imdb).

## Evaluation
1. Download the CNN Models for the UCF101 dataset, that are used in the journal, from [here](http://groups.inf.ed.ac.uk/hbilen-data/data/resnext50_dicnn.tar).
2. Choose the right model, split and input type (e.g.)
    ```matlab
    net = load('resnext50-rgb-arpool-split1.mat') ;
    net = dagnn.DagNN.loadobj(net) ;
    net.addLayer('errMC',ErrorMultiClass(),{'prediction','label'},'mcerr') ;
    opts.network = net ;
    opts.split = 1 ;
    opts.train.gpus = 1 ;
    opts.epochFactor = 0 ; 
    [net, info] = cnn_dicnn_rgb(opts)
    ```

## Citing Dynamic Image Networks

If you find the code useful, please cite:

        @inproceedings{Bilen2016a,
          author    = "Bilen, H. and Fernando, B. and Gavves, E. and Vedaldi, A. and Gould, S.",
          title     = "Dynamic Image Networks for Action Recognition",
          booktitle = "CVPR",
          year      = "2016"
        }
        @journal{Bilen2017a,
          author    = "Bilen, H. and Fernando, B. and Gavves, E. and Vedaldi, A.",
          title     = "Action Recognition with Dynamic Image Networks",
          journal   = " IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)",
          year      = "2017"
        }

## License
The analysis work performed with the program(s) must be non-proprietary work. Licensee and its contract users must be or be affiliated with an academic facility. Licensee may additionally permit individuals who are students at such academic facility to access and use the program(s). Such students will be considered contract users of licensee. The program(s) may not be used for commercial competitive analysis (such as benchmarking) or for any commercial activity, including consulting.

