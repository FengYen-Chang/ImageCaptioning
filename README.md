# Image Captioning

This topic demonstrates how to run the Image Caption sample application, which performs inference using image caption networks.

### Topology
The topology of this sample was forked from [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) which wrote by yunjey.

### Install Prerequirement

* For Ubuntu:
    ```sh
    pip install nltk pickle matplotlib pycocotools
    ```
* For windows:
    ```sh
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    pip install nltk pickle matplotlib
    ```

### How It Works 

Upon the start-up, the sample application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an caption for input image.

### Models

Download the converted __encoder__, __decoder__, and __word embedded__ model from [here](https://drive.google.com/drive/folders/1kjXg89lTO1jh9zTj8kJ8YrgKqATycbav?usp=sharing) and save it on path `{$PROJECT_ROOT}/models/`.

### Vocabulary

Download the vocabulary file from [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) which provide by original topology and save it on path `{$PROJECT_ROOT}/vocab`. In addition, the code for read vocabulary was using `build_vocab.py` which also provide by original topology.

### Usage

Using flags to assign the specific parameter:
```sh
-m_d            ->  assign the path of decoder model
-m_e            ->  assign the path of encoder model
-m_w            ->  assign the path of word embedded model
-v              ->  assign the vocabulary file
-i              ->  assign the path of input image
--cpu_extension ->  assign the path of cpu extension library
```

### Demo

Following the below commend to run the sample :
```sh
python infer.py -m_d ../models/decoder.xml -m_e ../models/encoder.xml -m_w ../models/embeded.xml -i ../images/example.png -v ../vocab/vocab.pkl --cpu_extension ${PATH_OF_CPU_EXTENSION_LIBRARY}
```
### Result

Inference image:

![alt text](images/example.png)

The caption for inference image:
__[<start> a group of giraffes standing next to each other . <end>]__