import src.models.unet_model
from src.utils import download_dataset, preprocess
import numpy as np
from scipy.misc import imread
import os
import matplotlib.pyplot as plt

def __init__(self):
    self.dataset_names = sorted([
        'neurofinder.00.00', 'neurofinder.00.01',
        'neurofinder.00.02', 'neurofinder.00.03',
        'neurofinder.00.04', 'neurofinder.00.05',
        'neurofinder.00.06', 'neurofinder.00.07', 'neurofinder.00.08',
        'neurofinder.00.09', 'neurofinder.00.10', 'neurofinder.00.11',
        'neurofinder.01.00', 'neurofinder.01.01', 'neurofinder.02.00',
        'neurofinder.02.01', 'neurofinder.03.00', 'neurofinder.04.00',
        'neurofinder.04.01', 'neurofinder.00.00.test', 'neurofinder.00.01.test',
        'neurofinder.01.00.test', 'neurofinder.01.01.test', 'neurofinder.02.00.test',
        'neurofinder.02.01.test', 'neurofinder.03.00.test', 'neurofinder.04.00.test',
        'neurofinder.04.01.test'])
    self.PATH = os.getcwd()
    self.datasets_dir = PATH + '/codeneuro_data'

def extract_roi(self):
    '''Extract Region of interest from the test data
    '''
    # Download and preprocess data if not already
    if not path.exists(self.datasets_dir):
        mkdir(datasets_dir)
        download_dataset().download_codeneuro()
        preprocess().prepare_codeneuro(show_images=1)

    train_images= []
    masks= []
    test_images= []

    for name in dataset_names:
        if '.test' not in name:
            img_path = datasets_dir +'/'+ name +'/images.tiff'
            mask_path = datasets_dir +'/'+ name +'/masks.tiff'
            train_images.append(np.array(imread(img_path)))
            masks.append(np.array(imread(mask_path)))
        else:
            test_img_path = datasets_dir +'/'+ name +'/images.tiff'
            test_images.append(np.array(imread(test_img_path)))

    # Save the list as np array
    input_ = np.asarray(train_images)
    labels = np.asarray(masks)
    test_ = np.asarray(test_images)

    # Resize them from (512,512) to (128,128)
    input_ = np.resize(input_, (128,128,len(train_images)))
    labels = np.resize(labels, (128,128,len(masks)))
    test_ = np.resize(test_, (128,128,len(test_images)))

    #Expand dimension to fit the UNet model
    input_ = np.expand_dims(input_, axis=0)
    labels = np.expand_dims(labels, axis=0)
    test_ = np.expand_dims(test_, axis=0)

    # Get the UNet model, compile and fit on training dataset
    model = unet(input_.shape)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.fit(input_, labels, batch_size=16, epochs=10, verbose=1)

    # get the prediction for test data
    pred = model.predict(test_)
    pred = np.resize(pred, (2,128,128))
    preds = pred.sum(axis=0)
    plt.imshow(preds, cmap='gray')
    # This shows the predicted neurons
    plt.show()

    # Here goes the code to extract ROI code and conver to json...
