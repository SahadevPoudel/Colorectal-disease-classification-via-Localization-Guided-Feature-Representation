from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

def load_kvasir_train_data(dataroot,batch_size):
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input
                                               )
        train_batches = train_datagen.flow_from_directory(dataroot+'training/',
                                                              target_size=(224, 224),
                                                              class_mode='categorical',
                                                              classes=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis'],
                                                              batch_size=batch_size)

        valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        valid_batches = valid_datagen.flow_from_directory(dataroot+'validation/',
                                                                  target_size=(224, 224),
                                                                  class_mode='categorical',
                                                                  classes=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis'],
                                                                  shuffle=False,
                                                                  batch_size=batch_size)
        return train_batches,valid_batches


def load_kvasir_test_data(dataroot, batch_size):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input
                                      )
    test_batches = test_datagen.flow_from_directory(dataroot + 'testing',
                                                    target_size=(224, 224),
                                                    class_mode='categorical',
                                                    classes=['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis','normal-cecum','normal-pylorus','normal-z-line','polyps','ulcerative-colitis'],
                                                    batch_size=batch_size)
    return test_batches
