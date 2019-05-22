import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, Embed, DecoderRNN2
from PIL import Image


# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
 
    # Define embed
    _embed = Embed(decoder.embed)
    # _decoder = DecoderRNN2(decoder.lstm, decoder.linear)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    
    sampled_ids = []
    state = (torch.zeros((1, 1, 512)).to(device), torch.zeros((1, 1, 512)).to(device))
    
    # sampled_ids = decoder(feature)
    # sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    inputs = feature    

    print (inputs.size())
    
    for i in range(20):
        pred, inputs, state = decoder(inputs, state)
        # _, pred = outputs.max(1)
        sampled_ids.append(pred)
        # print (pred)
        # inputs = _embed(pred)
        # print (inputs)
    print (state[0].size())
    print (state[1].size())
    print (np.array(state).shape)

    sampled_ids = torch.stack(sampled_ids, 1)
    sampled_ids = sampled_ids[0].cpu().numpy() 

    print (sampled_ids)

    # Save the model as .onnx format
    Decoder_ONNX_dir = '../models/onnx/decoder_nightly.onnx'
    Encoder_ONNX_dir = '../models/onnx/encoder.onnx'
    Embeded_ONNX_dir = '../models/onnx/embeded.onnx'

    state_for_onnx = torch.ones((1, 1, 512))

    # torch.onnx.export(encoder, image_tensor, Encoder_ONNX_dir)
    torch.onnx.export(decoder, (torch.ones(1, 256).to(device),state) , Decoder_ONNX_dir)
    # torch.onnx.export(_embed, pred, Embeded_ONNX_dir)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
