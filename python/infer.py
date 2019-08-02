import sys, os, time
import numpy as np
import argparse
import cv2

import timeit

from vocab import vocab 

from openvino.inference_engine import IENetwork, IEPlugin

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def parsing():
    parser = argparse.ArgumentParser(add_help = False)
    parser.add_argument('-m_d', '--model_decoder', default='', type=str)
    parser.add_argument('-m_e', '--model_encoder', default='', type=str)
    parser.add_argument('-i', '--input', default='', type=str)
    parser.add_argument('-t_l', '--max_length', default=20, type=int)
    parser.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)

    return parser

def loader(img_dir, size):
    ori_img = cv2.imread(img_dir)
    img = cv2.resize(ori_img, size, cv2.INTER_LANCZOS4)
    h, w, c = img.shape
    img = img.transpose((2, 0, 1)).reshape(1, c, h, w)

    return ori_img, img

def camera(cap, size):
    _, ori_img = cap.read()
    img = cv2.resize(ori_img, size, cv2.INTER_LANCZOS4)
    h, w, c = img.shape
    img = img.transpose((2, 0, 1)).reshape(1, c, h, w)

    return ori_img, img

def infer(image, 
          image_show, 
          exec_encoder, 
          encoder_input_blob, 
          encoder_output_blob, 
          exec_decoder, 
          decoder_input_blobs, 
          decoder_output_blobs, 
          state,
          MAX_LENGTH):
    encoder_input = {encoder_input_blob: image}
    features = exec_encoder.infer(encoder_input)   

    # d_inputs = np.zeros(decoder.inputs[decoder_input_blobs[0]].shape)
    # d_inputs[0] = features[encoder_output_blob]
    d_inputs = features[encoder_output_blob]

    sentence_ids = []

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

    sampled_caption = []
    for word_id in sentence_ids:
        word = vocab[int(word_id)]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print (sentence)    
    
    # parse sentence
    p_sentence = ' '.join(sampled_caption[1:-1])
    print (p_sentence)
    
    half_length = len(sampled_caption) // 2 + 1
    # p_sentence_show_1 = ' '.join(sampled_caption[1: half_length])
    # p_sentence_show_2 = ' '.join(sampled_caption[half_length: -1])
    
    # print (p_sentence_show_1, p_sentence_show_2)

    # show original pic
    # ori_img = cv2.imread(args.input)
    ori_h, ori_w, _ = image_show.shape
    # pt1 = (0, int(ori_h * 0.875))
    # pt2 = (ori_w, ori_h)
    # color = (255, 255, 255)
    # cv2.rectangle(image_show, pt1, pt2, color, -1)
    # cv2.putText(image_show, p_sentence_show_1, (0, int(ori_h * 0.9167)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2,cv2.LINE_AA) 
    # cv2.putText(image_show, p_sentence_show_2, (0, int(ori_h * 0.9767)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0),2,cv2.LINE_AA) 
   
    bg = np.zeros((972, 1920, 3), dtype=np.uint8)
    bg[:, :, 0] = np.mean(image_show[:, :, 0])
    bg[:, :, 1] = np.mean(image_show[:, :, 1])
    bg[:, :, 2] = np.mean(image_show[:, :, 2])

    ratio_x, ratio_y = ori_w / 1920, ori_h / 972
    print (ori_h, ori_w)
    print (ratio_x, ratio_y)
    
    if ratio_x > 1 or ratio_y > 1 :
        if ratio_x < ratio_y :
            image_show = cv2.resize(image_show, (1920, (int)(ori_h / ratio_x)), cv2.INTER_LANCZOS4)
            ori_h, ori_w, _ = image_show.shape
        else :
            image_show = cv2.resize(image_show, ((int)(ori_w / ratio_y), 972), cv2.INTER_LANCZOS4)
            ori_h, ori_w, _ = image_show.shape
    elif  ratio_x <= 1 and ratio_y <= 1:
        if ratio_x > ratio_y :
            image_show = cv2.resize(image_show, (1920, (int)(ori_h / ratio_x)), cv2.INTER_LANCZOS4)
            ori_h, ori_w, _ = image_show.shape
        else :
            image_show = cv2.resize(image_show, ((int)(ori_w / ratio_y), 972), cv2.INTER_LANCZOS4)
            ori_h, ori_w, _ = image_show.shape
        print (ori_h, ori_w)


    mv_h = 486 - ori_h // 2 
    mv_w = 960 - ori_w // 2

    bg[mv_h: (mv_h + ori_h), mv_w: (mv_w + ori_w), :] = image_show 

    text = np.zeros((108, 1920, 3), dtype=np.uint8)
    text[:] = 255
    cv2.putText(text, p_sentence, (50, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
    
    o_img = np.concatenate((bg, text), axis=0)

    # cv2.imwrite("output.jpg", o_img)
 
    # return image_show
    return o_img

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
    
    encoder_plugin = IEPlugin(device = 'GPU')
    exec_encoder = encoder_plugin.load(network = encoder)

    # init state -> zeros
    init_state_shape = decoder.inputs[decoder_input_blobs[1]].shape
    hidden_state = [np.zeros(init_state_shape, dtype=np.float32),
                    np.zeros(init_state_shape, dtype=np.float32)]

    # HETERO:CPU,GPU
    decoder_plugin = IEPlugin(device = 'CPU')
    decoder_plugin.add_cpu_extension(args.cpu_extension)
    # decoder_plugin.set_config({"GPU_EXTENSINO": "YES"})

    exec_decoder = decoder_plugin.load(network = decoder)
    
############################################################################
    MAX_LENGTH = args.max_length

    if args.input == 'webcam':
        cap = cv2.VideoCapture(0)
        while (True):
            image_show, image = camera(cap, (w, h))

            start = timeit.default_timer()
            show = infer(image, 
                        image_show,
                        exec_encoder, 
                        encoder_input_blob, 
                        encoder_output_blob, 
                        exec_decoder, 
                        decoder_input_blobs, 
                        decoder_output_blobs,
                        hidden_state, 
                        MAX_LENGTH)
            stop = timeit.default_timer()

            print ('Time: ', stop - start)

            hidden_state = [np.zeros(init_state_shape, dtype=np.float32),
                            np.zeros(init_state_shape, dtype=np.float32)]

            cv2.imshow("Image", show)
            k = cv2.waitKey(1)
            if k == 27:  # ESC
                cv2.destroyAllWindows()
                break
                
    elif args.input == 'image':
        import image_list
        idx = 0
        idx_ = [6, 8, 10, 11, 14, 17, 22, 25, 26, 27, 
                29, 30, 32, 33, 36, 37, 39, 40, 41, 43,
                44, 45, 46, 48, 50, 52, 59, 61, 62, 63, 
                65, 70, 71, 73, 78, 78, 79, 81, 86, 87, 
                93, 95, 97, 99, 105, 110, 117, 126, 128, 
                129, 131, 141, 156, 157, 159, 166, 167, 170, 
                172, 174, 175, 176, 183, 184, 185, 188, 190, 193, 
                195, 197, 199
                ]
        while (True):
            # print (idx_[idx % len(idx_)])
            image_show, image = loader(("../images/coco/" + image_list.image_list[idx_[idx % len(idx_)]]), (w, h))
            # image_show, image = loader(("../images/coco/" + image_list.image_list[idx % 200]), (w, h))            
            print (idx)
            start = timeit.default_timer()
            show = infer(image, 
                        image_show,
                        exec_encoder, 
                        encoder_input_blob, 
                        encoder_output_blob, 
                        exec_decoder, 
                        decoder_input_blobs, 
                        decoder_output_blobs,
                        hidden_state, 
                        MAX_LENGTH)
            stop = timeit.default_timer()

            print ('Time: ', stop - start)

            hidden_state = [np.zeros(init_state_shape, dtype=np.float32),
                            np.zeros(init_state_shape, dtype=np.float32)]
            idx += 1
            cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("Image", show)
            k = cv2.waitKey(2000)
            # k = cv2.waitKey(0)
            if k == 27:  # ESC
                cv2.destroyAllWindows()
                break
                # python infer.py -m_d ../models/IR/decoder_nightly.xml -m_e ../models/IR/encoder.xml -i image -t_l 20 --cpu_extension ../C++/build/ie_cpu_extension/libcpu_extension.so 

    else :
        image_show, image = loader(args.input, (w, h))

        start = timeit.default_timer()
        show = infer(image, 
                    image_show,
                    exec_encoder, 
                    encoder_input_blob, 
                    encoder_output_blob, 
                    exec_decoder, 
                    decoder_input_blobs, 
                    decoder_output_blobs,
                    hidden_state, 
                    MAX_LENGTH)
        stop = timeit.default_timer()

        print ('Time: ', stop - start)

        hidden_state = [np.zeros(init_state_shape, dtype=np.float32),
                        np.zeros(init_state_shape, dtype=np.float32)]

        cv2.imshow("Image", show)
        k = cv2.waitKey(0)
        if k == 27:  # ESC
            cv2.destroyAllWindows()
            
############################################################################

if "__main__" :
    main()
