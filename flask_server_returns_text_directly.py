import cv2
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Dropout
import keras.backend as K
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import math
import string
from spellchecker import SpellChecker
from flask import Flask, request, jsonify



line_img_array = []
char_list = string.ascii_letters + string.digits
spell = SpellChecker()

def sort_word(wordlist):
    wordlist.sort(key=lambda x: x[0])
    return wordlist

def preprocess_img(img, imgSize):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")

    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)
    most_freq_pixel = np.amax(img)
    target = np.ones([ht, wt]) * most_freq_pixel
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img

def pad_img(img):
    old_h, old_w = img.shape[0], img.shape[1]

    if old_h < 512:
        to_pad = np.ones((512 - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = 512
    else:
        to_pad = np.ones((roundup(old_h) - old_h, old_w)) * 255
        img = np.concatenate((img, to_pad))
        new_height = roundup(old_h)

    if old_w < 512:
        to_pad = np.ones((new_height, 512 - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = 512
    else:
        to_pad = np.ones((new_height, roundup(old_w) - old_w)) * 255
        img = np.concatenate((img, to_pad), axis=1)
        new_width = roundup(old_w) - old_w
    return img

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def unet(pretrained_weights=None, input_size=(512, 512, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def segment_into_lines(filename):
    
    img=cv2.imread(f'{filename}',0)
    ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    img=cv2.resize(img,(512,512))
   
    img= np.expand_dims(img,axis=-1)

    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    pred=np.squeeze(np.squeeze(pred,axis=0),axis=-1)

    

    coordinates=[]
    img = cv2.normalize(src=pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    ori_img=cv2.imread(f'{filename}',0)
 

    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)
    
    # Use try/except block to handle different versions of OpenCV
    try:
        # This is for OpenCV 4.x
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except ValueError:
        # This is for OpenCV 3.x
        _, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(ori_img, (int(x*rW), int(y*rH)), (int((x+w)*rW),int((y+h)*rH)), (255,0,0), 1)
        coordinates.append((int(x*rW),int(y*rH),int((x+w)*rW),int((y+h)*rH)))
    #cv2.imwrite("output.jpg",ori_img)

    for i in range(len(coordinates)-1,-1,-1):
        coors=coordinates[i]

        p_img=ori_img[coors[1]:coors[3],coors[0]:coors[2]].copy()

        line_img_array.append(p_img)

    return line_img_array

def save_and_display_lines(image_path, output_directory='listimages'):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Perform line segmentation
    line_img_array = segment_into_lines(image_path)

    # Save each segmented line and display it
    for idx, line_img in enumerate(line_img_array):
        output_filename = os.path.join(output_directory, f'Line_{idx + 1}.png')
        cv2.imwrite(output_filename, line_img)
        print(f"Saved: {output_filename}")


def segment_into_words(line_img, line_index):
    """This function takes in the line image and line index returns word images and the reference
    of the line they belong to."""
    img = pad_img(line_img)
    ori_img = img.copy()
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    
    img = cv2.resize(img, (512, 512))
    img = np.expand_dims(img, axis=-1)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    seg_pred = model2.predict(img)
    seg_pred = np.squeeze(np.squeeze(seg_pred, axis=0), axis=-1)
    seg_pred = cv2.normalize(src=seg_pred, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.threshold(seg_pred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, seg_pred)
    contours, hier = cv2.findContours(seg_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    (H, W) = ori_img.shape[:2]
    (newW, newH) = (512, 512)
    rW = W / float(newW)
    rH = H / float(newH)

    coordinates = []

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        coordinates.append((int(x*rW), int(y*rH), int((x+w)*rW), int((y+h)*rH)))

    coordinates = sort_word(coordinates)  # Sorting according to x-coordinates.
    word_counter = 0

    word_array = []
    line_indicator = []

    for (x1, y1, x2, y2) in coordinates:
        word_img = ori_img[y1:y2, x1:x2]
        word_img = preprocess_img(word_img, (128, 32))
        word_img = np.expand_dims(word_img, axis=-1)
        word_array.append(word_img)
        line_indicator.append(line_index)

    return line_indicator, word_array


def process_line_image(line_image_path, line_index):
    line_img = cv2.imread(line_image_path, 0)
    line_indicator, word_array = segment_into_words(line_img, line_index)
    return line_indicator, word_array


inputs = Input(shape=(32, 128, 1))

conv_1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
conv_2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
conv_3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv_3)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
conv_5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool_4)
batch_norm_5 = BatchNormalization()(conv_5)
conv_6 = Conv2D(512, (3,3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
conv_7 = Conv2D(512, (2,2), activation='relu')(pool_6)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)
outputs = Dense(len(char_list)+1, activation='softmax')(blstm_2)
act_model = Model(inputs, outputs)

act_model.load_weights('CRNN_model.hdf5')

def recognize_words(word_array):
    predictions = act_model.predict(word_array)
    out = K.get_value(K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                         greedy=True)[0][0])

    recognized_text = []
    for wordidxs in out:
        word = []
        for char in wordidxs:
            if int(char) != -1:
                word.append(char_list[int(char)])
        word = spell.correction(''.join(word))
        recognized_text.append(word)

    return recognized_text

def preprocess_image(img):
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def process_and_display_lines(image_path):
    # Perform line segmentation
    line_img_array = segment_into_lines(image_path)
    text=""
    # Process each line
    for idx, line_img in enumerate(line_img_array):
        print(f"Processing Line {idx + 1}")

        # Extract word images from the line
        line_indicator, word_array = segment_into_words(line_img, idx)

        # Recognize text in each word image
        recognized_text_batch = recognize_words(np.array(word_array))

        # Remove None values from recognized_text_batch
        recognized_text_batch = [text for text in recognized_text_batch if text is not None]

        # Display the original line image
        #cv2.imshow(f'Processed Line {idx + 1}', line_img)

        # Combine recognized texts to form the title
        title_text = ' '.join(recognized_text_batch)

        # Set the recognized text as the title of the window
        #cv2.setWindowTitle(f'Processed Line {idx + 1}', f'Recognized Text: {title_text}')
        print(title_text)
        # key = cv2.waitKey(0)   Wait until a key is pressed
        cv2.destroyAllWindows()
        text+=title_text
        # Break the loop if 'q' key is pressed
        # if key == ord('q'):
        #     break
    return text


model = unet()
model.load_weights('./text_seg_model.h5')
model2=unet()
model2.load_weights('./word_seg_model.h5')
image_path = './test_image.jpg'



app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image file from the request
    image_file = request.files['image']
    
    # Save the image temporarily
    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Perform image processing and get the recognized text
    text = process_and_display_lines(image_path)

    # Return the recognized text as JSON response
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

#process_and_display_lines(image_path)
