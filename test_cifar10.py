from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt

def norm_pix(img):
    # Scales pixel values of a vector of images to the range [0,1]
    img_norm = img.astype('float32')
    img_norm = img_norm / 255.0
    return img_norm

def test_cifar10(model, n_epochs=100, n_batch_size=64, save_diag=False):
    #==================================================================
    # Trains a given model on CIFAR-10 and evaluates the fit.
    #
    # Arguments:
    #   model (Sequential): the model to evaluate
    #   n_epochs (integer): number of passes over the training data
    #   n_batch_size (integer): number of samples per training pass
    #   save_diag (Boolean): whether or not to save the plots to file
    #
    # Returns: trained model
    #==================================================================
    (trainImg, trainLab), (testImg, testLab) = cifar10.load_data()
    
    # Preprocess data for model training (normalize/one-hot encoding)
    trainImg = norm_pix(trainImg)
    testImg = norm_pix(testImg)
    trainLab = to_categorical(trainLab)
    testLab = to_categorical(testLab)
    
    # Train model
    history = model.fit(trainImg,
                        trainLab,
                        epochs=n_epochs,
                        batch_size=n_batch_size,
                        validation_data=(testImg, testLab))
    
    # Training diagnostics
    end_index = n_epochs - 1
    end_fit = [history.history['accuracy'][end_index],
               history.history['loss'][end_index],
               history.history['val_accuracy'][end_index],
               history.history['val_loss'][end_index]]
    
    print('Model training diagnostics with %d epochs of batch size %d:' % (n_epochs, n_batch_size))
    print('--------------------------------------------------------------------------------------')
    print('Training accuracy: %f' % end_fit[0])
    print('Training loss: %f' % end_fit[1])
    print('Test accuracy: %f' % end_fit[2])
    print('Test loss: %f' % end_fit[3])
    
    # Plots
    fig, (ax1, ax2) = plt.subplots(2)
    
    ax1.set_title('Classification Accuracy')
    ax1.plot(history.history['accuracy'], color='black', label='Training')
    ax1.plot(history.history['val_accuracy'], color='red', label='Test')
    ax1.legend()
    
    ax2.set_title('Loss')
    ax2.plot(history.history['loss'], color='black', label='Training')
    ax2.plot(history.history['val_loss'], color='red', label='Test')
    ax2.legend()
    
    if save_diag:
        fig.savefig('diag.png')
    
    return model