# AOSSig Dataset

This is the official repository for AOSSig dataset, associated with the paper: ***A New Multi-task Chinese Handwritten Signature Document Dataset with pixel level annotation***.

## Catalog

1. [UsageDownload](#Usage&Download)
2. [Description](#Description)
3. [Collection](#Collection)
4. [Directory-Format](#Directory-Format)
5. [Experimental-Result](#Experimental-Result)
6. [Data-Synthesis](#Data-Synthesis)
7. [Traning&Eval](#training&eval)
8. [License](#License)
9. [Copyright](#Copyright)

## Usage&Download

* Usage
  * The AOSSig dataset can only be used for non-commercial research purposes. For scholar or organization who wants to use the AOSSig dataset, please first fill in this [Application Form](./Application_Form/Application-Form-for-Using-AOSSig_Chinese.docx) and sign the [Legal Commitment](./Application_Form/Legal-Commitment_Chinese.docx) and email them to us. When submitting the application form to us, please list or attached 1-2 of your publications in the recent 6 years to indicate that you (or your team) do research in the related research fields of handwriting verification, handwriting analysis and recognition, document image processing, and so on. 
  * All users must follow all use conditions; otherwise, the authorization will be revoked.
* Download
  You can download the dataset by following this links:  [Baidu Yun](https://pan.baidu.com/share/init?surl=8KTsYGlKkFMJGlmdHG_2sQ) | [Google Drive](https://drive.google.com/drive/folders/1-n_aqhI7BLCh6HG_vgp5FlXtNLr-nS1q?usp=sharing). We will give you the file extraction password after your application has been received and approved.

## Description

AOSSig dataset is a real-scene Chinese handwritten signature documents dataset, which caters to various tasks such as signature detection, segmentation, and recognition, and consists of three subsets: AOSS4000, AOSSDoc, and AOSSynDoc.

* AOSSig4000 is a real-world Chinese handwritten signature document dataset with detection and segmentation labels, consisting of 3,500 text signatures for training and 500 for testing.
* AOSDoc composes 3,182 document templates with different noise levels, all of which signature locations are annotated. The annotation facilitates the synthesis of data as close as to real scenarios.
* AOSSynDoc dataset is a synthetic dataset obtained by using Brute-force pasting, Poisson fusion, and Text renderer algorithms, which can be used for pre-training and other work in downstream tasks.

## Collection

* AOSSig4000
  We collected 951 document templates from the internet, constructed and color-printed 5,000 paper blank Chinese document templates for 150 volunteers to sign in. Among them, various noises including fingerprints and stamps were randomly added to the templates. Volunteers were given the freedom to choose ink colors of blue, red, and black at random while all signature names were generated according to Chinese naming traditions. During the data collection process, such as ink leakage or lack of ink in the signature pen, errors caused by improper operation of the signatory, or unrealistic scribbles, factors are inevitable. Such situations often led to incomplete signatures or serious adhesion. Signatures could not meet the actual collection needs, so we manually screened out such data. As a result, 4,000 documents signed by 108 volunteers were photographed or scanned into JPEG image format, and we split the training and test sets with complete independence of volunteers as the principle. The collection process of AOSSig4000 is blind to any personal identity information and can be safely disclosed to the public. Firstly, templates collected from the internet do not contain any sensitive information; secondly, contents of signatures are not volunteers' real names; thirdly, seals are customized for real scenes and their contents are marked for data collection and testing purposes only; and finally, all fingerprints come from merely five volunteers and do not match with the signatures.
* AOSSDoc
  The AOSSDoc is built based on 951 document templates obtained in AOSSig4000. By manually adding a variety of noises referring to the difficult definition of L1 ~L5, constructed 3182 blank Chinese templates with different styles. Then, we proceed to mark the position box for several signatures on the templates using the labelme tool, which supports subsequent developers to generate corresponding signatures at the correct signature positions.
* AOSSynDoc
  The data of AOSSynDoc was acquired by using Brute-force pasting, Poisson fusion and Text renderer, based on the Chinese document template AOSDoc dataset and two handwritten Chinese signature datasets Chisig and MSDS.

## Directory-Format

* AOSSig4000

  ```
  ├─AOSSig4000
  │  ├─L1
  │  │  ├─data
  │  │  │  ├─training
  │  │  │  │  ├─train_L1_ photo_black_nothing_00001.jpg
  │  │  │  │  ├─train_L1_scan_blue_nothing_00002.jpg
  │  │  │  │  ├─...
  │  │  │  ├─test
  │  │  │  │  ├─test_L1 photo_black_nothing_00001_writer001.jpg
  │  │  │  │  ├─test_L1_scan_blue_nothing_00002_writer002.jpg
  │  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─training
  │  │  │  │  ├─train_L1_ photo_black_nothing_00001.txt
  │  │  │  │  ├─train_L1_scan_blue_nothing_00002.txt
  │  │  │  │  ├─...
  │  │  │  ├─test
  │  │  │  │  ├─test_L1 photo_black_nothing_00001_writer001.txt
  │  │  │  │  ├─test_L1_scan_blue_nothing_00002_writer002.txt
  │  │  │  │  ├─...
  │  │  ├─anno_mask
  │  │  │  ├─training
  │  │  │  │  ├─train_L1_ photo_black_nothing_00001.png
  │  │  │  │  ├─train_L1_scan_blue_nothing_00002.png
  │  │  │  │  ├─...
  │  │  │  ├─test
  │  │  │  │  ├─test_L1 photo_black_nothing_00001_writer001.png
  │  │  │  │  ├─test_L1_scan_blue_nothing_00002_writer002.png
  │  │  │  │  ├─...
  │  ├─L2
  │  │  ├─data
  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─...
  │  │  ├─anno_mask
  │  │  │  ├─...
  │  ├─L3
  │  │  ├─data
  │  │  │  ├─...

  ```

  * The AOSSig4000 folder contains five subfolders, each of them is split into three parts which named data, anno_label and anno_mask respectively.
  * The document images in data are saved in the JPEG format, the pixel-level annotated images in anno_mask are saved in the PNG format,  the texts in anno_label are saved in the TXT format.
  * The naming of each file follows the same format: flag_level_device_color_noise_index_writer.
    * The flag is train or test, train indicate that this file is used for training, while test
      indicates that it is a test sample,
    * The level indicates the type of data difficulty of levels,
    * The device indicates the device type of data collection, which can be either a mobile phone or a high-speed camera, represented by photo and scan respectively,
    * The color indicates the color of the signature pen, including black, red, and blue,
    * The noise indicates the type of data interference, which includes nothing, fingerprint, stamp, underline, printed font, degradation, and others. If a certain data has composite noise, it will be connected with the symbol *,
    * The index indicates the number of this file (.jpg or .txt or .png),
    * The writer represents the signatory category information but only test sets include this attribute label and have not been added to the training set for the time being.
      
* AOSDoc

  ```
  ├─AOSDoc
  │  ├─L1
  │  │  ├─data
  │  │  │  ├─Templates_L1_00001.jpg
  │  │  │  ├─Templates_L1_00002.jpg
  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─Templates_L1_00001.txt
  │  │  │  ├─Templates_L1_00002.txt
  │  │  │  ├─...
  │  ├─L2
  │  │  ├─data
  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─...
  ```

  * The AOSDoc folder contains five subfolders, each of them is split into two parts which named data, anno_label respectively.
  * The document images in data are saved in the JPEG format,  the texts in anno_label are saved in the TXT format.
  * The naming of each file follows the same format: flag_level_index
    * The flag is unified and named as Templates.
    * The level indicates the difficulty level of the template.
    * The index indicates the number of this file (.jpg or .txt).
* AOSSynDoc

  ```
  ├─Bruce-force_syn_data
  │  ├─L1
  │  │  ├─data
  │  │  │  ├─L1_00001.jpg
  │  │  │  ├─L1_00002.jpg
  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─L1_00001.txt
  │  │  │  ├─L1_00002.txt
  │  │  │  ├─...
  │  │  ├─anno_mask
  │  │  │  ├─L1_00001.png
  │  │  │  ├─L1_00002.png
  │  │  │  ├─...
  │  ├─L2
  │  │  ├─data
  │  │  │  ├─...
  │  │  ├─anno_label
  │  │  │  ├─...
  │  │  ├─anno_mask
  │  │  │  ├─...

  ```

  * The directory Layout and naming of each file in AOSSynDoc refer to AOSSig4000.

## Experimental-Result

* Detection Benchmark
  <p><img src='images\benchmark\detection_benchmark.png', align='center'/></p>
  Experimental results show that all detection methods achieve good results148 on AOSSig4000. The performance gap between segmentation-based detection methods and anchor-based detection methods is insignificant. In terms of IoU ≥ 0.5 evaluation metric, PSENet achieves the best performance (F = 0.966) compared to other models and score better on our own dataset than on scene text datasets such as Total-Text (F = 0.809) and CTW1500 (F = 0.822). DETR based on Transformer architecture has good performance under IoU ≥ 0.75.
* Segmentation Benchmark

  <p><img src='images\benchmark\segmentation_benchmark.png', align='center'/></p>
  Experimental results show that Semantic segmentation models based on deep convolution architectures perform better in overall, with SegNeXt achieving the highest scores in all three metrics. Image generation-based model CycleGAN performs similarly as deep convolution-based segmentation models. However, transformer-based models perform the worst, with fgIoU ranging from 0.43 to 0.61.

## Data-Synthesis

* Prepare or download the [AOSDoc](https://github.com/open-aoslab/AOSSig) and [Chisig](https://github.com/dskezju/ChiSig) datasets
* Modify the configuration file
  ```
  vim ./sig_gen/config/config.yaml
  data:
    background:
        data_dir: ''      # the dir of templeate images in AOSDoc
        data_save_dir: '' # the save dir of image synthesized 
    signature:
        sig_img_dir: ''   # the dir of handwriting signature in chisig
        sig_mask_dir: ''  # the dir of handwriting signature mask in chisig
        sig_loc_dir: ''   # the dir of location labels in AOSDoc  
        mask_save_dir: '' # the save dir of signature mask label
        anno_save_dir: '' # the save dir of signature position label 
  compose:
    ...
    funsion: ''           # supporting funsion methods: possion、violence、text_render
  ```
* Perform data generation
  ```
  python ./sig_gen/core/sig_generation.py
  ```

## Training&Eval

* Recommended environment

  ```
      python 3.7+ 
      pytorch 1.9.0
      torchvision 0.10.0
      opencv-python 4.5.3.56
      imgaug 0.4.0
  ```
* Training

  * Segmentaion
    In this section, the source code of partial methods implemented by ourself were provided. The training process is as follows:
    ```
    cd ./training/SigSeg
    python training.py -m [mode_name] -d [data_root]
    ```
  * Detection
    Please refer to this respository [PSENet](https://github.com/whai362/PSENet)
* Eval

  * Segmentation
    ```
    python eval_lib/seg_eval/seg_eval.py -p [pred_root] -g [gt_root]
    ```
  * Detection
    ```
    python eval_lib/detect_eval/detect_eval.py -p [pred_root] -g [gt_root]
    ```

## License

AOSSig Dataset should be used and distributed under [Creative Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes

## Copyright

For commercial purpose usage, please contact Chongqing Handwriting Big Data Research Institute:[qinxunhui@aosign.cn]()
