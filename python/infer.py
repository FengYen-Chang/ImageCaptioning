import sys, os, time
import numpy as np
import argparse
import cv2

from openvino.inference_engine import IENetwork, IEPlugin

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def parsing():
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('-m_d', '--model_decoder', default='', type=str)
    parser.add_argument('-m_e', '--model_encoder', default='', type=str)
    parser.add_argument('-m_w', '--model_embedded', default='', type=str)
    parser.add_argument('-v', '--vocab', default='', type=str)
    parser.add_argument('-i', '--input', default='', type=str)
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
    MAX_LENGTH = 20

##########################################################################

    encoder_graph = args.model_encoder
    encoder_weight = args.model_encoder[:-3] + 'bin'

    encoder = IENetwork(model = encoder_graph,
                        weights = encoder_weight)

    decoder_graph = args.model_decoder
    decoder_weight = args.model_decoder[:-3] + 'bin'

    decoder = IENetwork(model = decoder_graph,
                        weights = decoder_weight)
    
    embedded_graph = args.model_embedded
    embedded_weight = args.model_embedded[:-3] + 'bin'

    embedded = IENetwork(model = embedded_graph, 
                         weights = embedded_weight)

############################################################################

    iter_encoder_inputs = iter(encoder.inputs)
    iter_encoder_outputs = iter(encoder.outputs)

    iter_decoder_inputs = iter(decoder.inputs)
    iter_decoder_outputs = iter(decoder.outputs)
    
    iter_embedded_inputs = iter(embedded.inputs)
    iter_embedded_outputs = iter(embedded.outputs)

    encoder_input_blob = next(iter_encoder_inputs)
    encoder_output_blob = next(iter_encoder_outputs)
    
    decoder_input_blobs = []
    decoder_output_blobs = []
    
    for i in iter_decoder_inputs:
        decoder_input_blobs.append(i)
        print (i)

    for i in iter_decoder_outputs:
        decoder_output_blobs.append(i)
        print (i)

    embedded_input_blob = next(iter_embedded_inputs)
    embedded_output_blob = next(iter_embedded_outputs)

###########################################################################

    _, c, h, w = encoder.inputs[encoder_input_blob].shape
    image = loader(args.input, (w, h))
    
    encoder_plugin = IEPlugin(device = 'CPU')
    exec_encoder = encoder_plugin.load(network = encoder)

    # init state -> zeros
    init_state_shape = decoder.inputs[decoder_input_blobs[1]].shape
    print (init_state_shape)
    print (decoder.inputs[decoder_input_blobs[0]].shape)
    state = [np.zeros(init_state_shape, dtype=np.float32),
             np.zeros(init_state_shape, dtype=np.float32)]

    decoder_plugin = IEPlugin(device = 'CPU')
    exec_decoder = decoder_plugin.load(network = decoder)

    embedded_plugin = IEPlugin(device = 'CPU')
    embedded_plugin.add_cpu_extension('/home/john-server/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so')
    exec_embedded = embedded_plugin.load(network = embedded)

############################################################################

    encoder_input = {encoder_input_blob: image}
    features = exec_encoder.infer(encoder_input)   

    d_inputs = np.zeros(decoder.inputs[decoder_input_blobs[0]].shape)
    d_inputs[0] = features[encoder_output_blob]

    sentance_ids = []
 
    for i in range(1):
        print (i)
        if i == 0:
            decoder_inputs = {decoder_input_blobs[0]: d_inputs,
                              decoder_input_blobs[1]: state[0], 
                              decoder_input_blobs[2]: state[1]}
    
        else: 
            decoder_inputs = {decoder_input_blobs[0]: embedded_input,
                              decoder_input_blobs[1]: state[0],
                              decoder_input_blobs[2]: state[1]}

        decoder_out = exec_decoder.infer(decoder_input)
        state = [decoder_out[decoder_output_blobs[0]], decoder_out[decoder_output_blobs[1]]]
        pred = np.argmax(out[decoder_output_blobs[2]], 1)
        sentance_ids.append(pred)
        embedded_input = exec_embedded(pred)

    print (sampled_ids)
        
    
if "__main__" :
    main()
