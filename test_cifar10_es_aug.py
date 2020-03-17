from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from math import ceil
from keras.models import load_model
from matplotlib import pyplot as plt

def norm_pix(img):
    # Scales pixel values of a vector of images to the range [0,1]
    img_norm = img.astype('float32')
    img_norm = img_norm / 255.0
    return img_norm

def test_cifar10(model, n_epochs=50000, n_batch_size=64, n_patience=10, save='fitted_model'):
    #==================================================================
    # Trains a given model on CIFAR-10 and evaluates the fit.
    #
    # Arguments:
    #   model (Sequential): the model to evaluate
    #   n_epochs (integer): number of passes over the training data
    #   n_batch_size (integer): number of samples per training pass
    #   save (string): filename for saved diagnostic plots
    #
    # Returns: Fit history
    #==================================================================
    (trainImg, trainLab), (testImg, testLab) = cifar10.load_data()
    
    # Normalize/encode data (normalize/one-hot encoding)
    trainImg = norm_pix(trainImg)
    testImg = norm_pix(testImg)
    trainLab = to_categorical(trainLab)
    testLab = to_categorical(testLab)
    
    # Instantiate data augmentation
    data_gen = ImageDataGenerator(width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  rotation_range=90,
                                  horizontal_flip=True)
    data_gen.fit(trainImg)
    
    # Train model
    history = model.fit_generator(data_gen.flow(trainImg, trainLab, batch_size=n_batch_size),
                                steps_per_epoch=ceil(50000/n_batch_size),
                                epochs=n_epochs,
                                callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=n_patience, verbose=1),
                                           ModelCheckpoint(save + '.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)],
                                           validation_data=(testImg, testLab))
    
    # Training diagnostics
    fitted_model = load_model(save + '.h5')
    _, train_acc = fitted_model.evaluate(trainImg, trainLab, verbose=0)
    _, test_acc = fitted_model.evaluate(testImg, testLab, verbose=0)
    print('Best model performance:')
    print('Training set accuracy: %.4f' % train_acc)
    print('Test set accuracy: %.4f' % test_acc)
    
    # Plots of fit
    hist_train_acc = history.history['accuracy']
    hist_train_loss = history.history['loss']
    hist_test_acc = history.history['val_accuracy']
    hist_test_loss = history.history['val_loss']
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    x_range = range(1, len(hist_train_acc) + 1)
    
    ax1.set_title('Classification Accuracy')
    ax1.plot(x_range, hist_train_acc, color='black', label='Training')
    ax1.plot(x_range, hist_test_acc, color='red', label='Test')
    ax1.axvline(len(hist_train_acc) - 10, color='blue', label='Best')
    ax1.legend()
    
    ax2.set_title('Cross Entropy Loss')
    ax2.plot(x_range, hist_train_loss, color='black', label='Training')
    ax2.plot(x_range, hist_test_loss, color='red', label='Test')
    ax2.axvline(len(hist_train_acc) - 10, color='blue', label='Best')
    ax2.legend()
   
    # Save graphics
    model_filename = 'model_' + save + '.png'
    diag_filename = 'diag_' + save + '.png'
    
    plot_model(model, to_file=model_filename)
    fig.savefig(diag_filename)
    
    return model