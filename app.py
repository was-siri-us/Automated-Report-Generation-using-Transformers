
import tensorflow as tf
from CNN_encoder import CNN_Encoder
from gpt2.gpt2_model import TFGPT2LMHeadModel
from tokenizer_wrapper import TokenizerWrapper
import numpy as np
from PIL import Image
from skimage.transform import resize
import re


from keras_preprocessing.image import img_to_array,load_img
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import os
import base64
from io import BytesIO



app=Flask("__name__")


# UPLOAD_FOLDER = "uploads"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
# CORS(app, supports_credentials=True) 

@app.route('/', methods=['POST'])

def hello():
    print("Hello")



@app.route("/predict",methods=["POST"])
def predict():
    try:
        print("hello im app start")
        print("hello")
        
        
        input_data = request.get_json()
        image_data = input_data['image']
        # filename = input_data.get('name', 'default_filename.jpg')
        
        # l1 = filename
        # actual_image = l1[:-4]
        

        # Ensure the image data is a bytes-like object
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to image
        image = Image.open(BytesIO(image_bytes))

        #MIGHT NEED CHANGES HERE -------------------------->
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (224,224))
        images = np.asarray([image_array])

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


        # pattern = re.compile(r'"([^"]*)"')
        # match = pattern.search(sentence)
        # if match:
        #     pred = match.group(1)
        # else:
        #     pred="No sentence generated"

        
        response_data = {
            'prediction':sentence,
        }       
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__=="__main__":
    print("server started raa chaari")
    tokenizer_wrapper = TokenizerWrapper(200, 1001)

    encoder = CNN_Encoder('pretrained_visual_model', 'fine_tuned_chexnet', 2,
                          encoder_layers=[0.4], tags_threshold=-1, num_tags=105)

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, './ckpts/CDGPT2/', max_to_keep=1)
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))








    # app.run(port=5000,debug=True)
    app.run(host='0.0.0.0',port=80,debug=False)
    
