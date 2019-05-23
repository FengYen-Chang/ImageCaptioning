import sys, os, time
import numpy as np
import argparse
import cv2

import pickle # for vocabulary
from build_vocab import Vocabulary

from openvino.inference_engine import IENetwork, IEPlugin

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def parsing():
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('-m_d', '--model_decoder', default='', type=str)
    parser.add_argument('-m_e', '--model_encoder', default='', type=str)
    parser.add_argument('-v', '--vocab', default='', type=str)
    parser.add_argument('-i', '--input', default='', type=str)
    parser.add_argument('-t_l', '--max_length', default=20, type=int)
    parser.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)

    return parser

def loader(img_dir, size):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, size, cv2.INTER_LANCZOS4)
    h, w, c = img.shape
    img = img.transpose((2, 0, 1)).reshape(1, c, h, w)

    return img

def main() :
    args = parsing().parse_args()

##########################################################################

    encoder_graph = args.model_encoder
    encoder_weight = args.model_encoder[:-3] + 'bin'

    encoder = IENetwork(model = encoder_graph,
                        weights = encoder_weight)

    decoder_graph = args.model_decoder
    decoder_weight = args.model_decoder[:-3] + 'bin'

    decoder = IENetwork(model = decoder_graph,
                        weights = decoder_weight)
    decoder.add_outputs('60')

############################################################################

    iter_encoder_inputs = iter(encoder.inputs)
    iter_encoder_outputs = iter(encoder.outputs)

    iter_decoder_inputs = iter(decoder.inputs)
    iter_decoder_outputs = iter(decoder.outputs)
    

    encoder_input_blob = next(iter_encoder_inputs)
    encoder_output_blob = next(iter_encoder_outputs)
    
    decoder_input_blobs = []
    decoder_output_blobs = []
    
    for i in iter_decoder_inputs:
        decoder_input_blobs.append(i)

    for i in iter_decoder_outputs:
        decoder_output_blobs.append(i)

###########################################################################

    _, c, h, w = encoder.inputs[encoder_input_blob].shape
    image = loader(args.input, (w, h))
    
    encoder_plugin = IEPlugin(device = 'CPU')
    exec_encoder = encoder_plugin.load(network = encoder)

    # init state -> zeros
    init_state_shape = decoder.inputs[decoder_input_blobs[1]].shape
    state = [np.zeros(init_state_shape, dtype=np.float32),
             np.zeros(init_state_shape, dtype=np.float32)]

    decoder_plugin = IEPlugin(device = 'CPU')
    decoder_plugin.add_cpu_extension(args.cpu_extension)
    exec_decoder = decoder_plugin.load(network = decoder)
    
############################################################################

    encoder_input = {encoder_input_blob: image}
    features = exec_encoder.infer(encoder_input)   

    d_inputs = np.zeros(decoder.inputs[decoder_input_blobs[0]].shape)
    d_inputs[0] = features[encoder_output_blob]
    d_inputs = features[encoder_output_blob]

    sentence_ids = []
    MAX_LENGTH = args.max_length

    for i in range(MAX_LENGTH):
        decoder_inputs = {decoder_input_blobs[0]: d_inputs,
                          decoder_input_blobs[1]: state[0], 
                          decoder_input_blobs[2]: state[1]}

        decoder_out = exec_decoder.infer(decoder_inputs)

        state[0] = decoder_out[decoder_output_blobs[0]].copy()
        state[1] = decoder_out[decoder_output_blobs[1]].copy()
        sentence_ids.append(decoder_out[decoder_output_blobs[3]][0])
        d_inputs = decoder_out[decoder_output_blobs[2]].copy()
        
        if (sentence_ids[-1] == 2):
            break 
        
    print (sentence_ids)

    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)

    sampled_caption = []
    for word_id in sentence_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print (sentence)    
    
    # parse sentence
    p_sentence = ' '.join(sampled_caption[1:-1])
    print (p_sentence)   
 
    # show original pic
    ori_img = cv2.imread(args.input)
    cv2.imshow("ori img", ori_img)
    k = cv2.waitKey(0)
    if k == 27:  # ESC
        cv2.destroyAllWindows()
 
if "__main__" :
    main()
