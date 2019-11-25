import argparse
import sys
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing import image
import numpy as np
from Colon_load_data import *
from model import *
from Kvasir_load_data import *
import cv2

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default='/home/poudelas/Documents/Colorectaldata/', help='Directory to dataset')
    parser.add_argument('--mode', type=str, default='train', help='train/test/visualize')
    parser.add_argument('--checkckpt_dir', type=str, default='/home/poudelas/Documents/Colorectaldata/colonmodel/checkpoint_dir', help='checkpont logs')
    parser.add_argument('--plot', type=str, default='yes', help='yes/no')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr',type=int,default=0.001,help='Learning rate')
    parser.add_argument('--image_path',type=str,help='Enter image path you want to visualize')
    parser.add_argument('--load_model',type=str,help='Load the saved model')
    parser.add_argument('dataset_type',type=str,default='Colorectal',help='Colorectal | KSAVIR')

    return parser.parse_args(argv)

def main(args):
    if args.mode == 'train':
        if args.dataset_type == 'Colorectal':
            train_batches, valid_batches = load_colorectal_train_data(args.dataroot, args.batch_size)
            model = ResNet50(classes=5)
        else:
            train_batches, valid_batches = load_kvasir_train_data(args.dataroot, args.batch_size)
            model = ResNet50(classes=8)
        model.compile(SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True),loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint(args.checkckpt_dir+'/train-{epoch:05d}--{val_loss:.4f}.h5', verbose=1,monitor='val_loss', save_best_only=True, mode='auto')
        model.fit_generator(train_batches,
                            steps_per_epoch=train_batches.samples // args.batch_size,
                            validation_data=valid_batches,
                            validation_steps=valid_batches.samples // args.batch_size,
                            epochs=args.epoch,
                            workers=10,
                            callbacks=[checkpoint]
                            )
        model.save(args.checkckpt_dir+'/final_model.h5')
    if args.mode == 'test':
        if args.dataset_type == 'Colorectal':
            test_batches = load_colon_test_data(args.dataroot, args.batch_size)
        else:
            test_batches = load_kvasir_test_data(args.dataroot,args.batch_size)
        testing_model = load_model(args.load_model)
        results = testing_model.evaluate_generator(generator=test_batches)
        print('Testing accuracy: '+ results[0])
        print('Testing Loss: '+ results[1])

    if args.mode =='visualize':
        model = load_model(args.load_model)
        img_path = args.image_path
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor1 = np.expand_dims(img_tensor, axis=0)
        img_tensor2 = preprocess_input(img_tensor1)
        # for i, layer in enumerate(model.layers):
        #     print(i, layer.name
        preds = model.predict(img_tensor2)
        print(np.argmax(preds[0]))
        uc = model.output[:, np.argmax(preds[0])]
        last_conv_layer = model.get_layer('res5c_branch2c')
        grads = K.gradients(uc, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor2])
        for i in range(2048):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.3 + img
        cv2.imwrite('1.jpg', superimposed_img)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
