
from flask import Flask, request, render_template,jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

import tensorflow as tf
from CNN_encoder import CNN_Encoder
from gpt2.gpt2_model import TFGPT2LMHeadModel
from tokenizer_wrapper import TokenizerWrapper
import numpy as np
from skimage.transform import resize
import re
import os






app = Flask(__name__)
CORS(app)


tokenizer_wrapper = None
encoder = None
decoder = None
ckpt_manager = None

def initialize_models():
    global tokenizer_wrapper, encoder, decoder, ckpt_manager

    print("Initializing models...")
    if(encoder == None):
        tokenizer_wrapper = TokenizerWrapper(200, 1001)
        encoder = CNN_Encoder('pretrained_visual_model', 'fine_tuned_chexnet', 2,
                            encoder_layers=[0.4], tags_threshold=-1, num_tags=105)
        decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)
        optimizer = tf.keras.optimizers.Adam()
        ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, './ckpts/CDGPT2/', max_to_keep=1)

        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))

# Initialize models when the server starts
initialize_models()







@app.route('/')
def index():
    return "<h1>Hello World</h1>"

@app.route('/upload', methods=['POST'])
def upload():
    global encoder,decoder,optimizer,ckpt,ckpt_manager
    if 'image' in request.files:
        image_file = request.files['image']
        image_data = image_file.read()

        # Open the image using Pillow
        image = Image.open(BytesIO(image_data))

        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (224,224))
        images = np.asarray([image_array])

        # Process the image (you can perform additional processing here)


        visual_features, tags_embeddings = encoder(images)
        dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
        num_beams = 7
        visual_features = tf.tile(visual_features, [num_beams, 1, 1])
        tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])




        tokens = decoder.generate(dec_input, max_length=200, num_beams=num_beams, min_length=3,
                                    eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), no_repeat_ngram_size=0,
                                    visual_features=visual_features,
                                    tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)
        sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
        sentence = tokenizer_wrapper.filter_special_words(sentence)





        # Return the image along with its size as JSON
        response_data = {
            'Predicted': sentence,
        }

        return jsonify(response_data)
    else:
        return 'No file provided.'

if __name__ == '__main__':
    app.run(debug=True)
