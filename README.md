# Image Captioning

This topic demonstrates how to run the Image Caption sample application, which performs inference using image caption networks.

### Topology
The topology of this sample was forked from [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) which wrote by yunjey.

### How It Works 

Upon the start-up, the sample application reads command line parameters and loads a network and an image to the Inference Engine plugin. When inference is done, the application creates an caption for input image.

### Models

Download the converted __encoder__, __decoder__, and __embedded__ model from [here]() and it to file `models/IR`

### Vocabulary

Download the vocabulary file from [here]() which provide by original topology and save it to file `vocab`. In addition, the code for read vocabulary was using `build_vocab.py` which also provide by original topology.

### Usage

Following the below commend to run the sample :
```
python infer.py -m_d ../models/IR/decoder.xml -m_e ../models/IR/encoder.xml -m_w ../models/IR/embeded.xml -i ../images/example.png -v ../vocab/vocab.pkl
```

