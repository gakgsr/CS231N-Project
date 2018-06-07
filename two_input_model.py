import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras import regularizers, optimizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, concatenate, Maximum
from keras import applications
from keras.utils.np_utils import to_categorical
import math
import cv2
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

epochs = 10
batch_size = 16



train_labels = np.load('classes_train.npy')
class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
num_classes = len(np.unique(train_labels))

train_data_xception = np.load('bottleneck_features_train_xception.npy')
train_data_vgg = np.load('bottleneck_features_train.npy')

train_labels = to_categorical(train_labels, num_classes=num_classes)

validation_data_xception = np.load('bottleneck_features_validation_xception.npy')
validation_data_vgg = np.load('bottleneck_features_validation.npy')

test_data_xception = np.load('bottleneck_features_test_xception.npy')
test_data_vgg = np.load('bottleneck_features_test.npy')

validation_labels2 = np.load('classes_valid.npy')
validation_labels = to_categorical(validation_labels2, num_classes=num_classes)

inputs_xception = Input(shape=train_data_xception.shape[1:], name='xception_input')
inputs_vgg = Input(shape=train_data_vgg.shape[1:], name='vgg_input')
x5 = Flatten()(inputs_xception)
x6 = Flatten()(inputs_vgg)
x1 = Dense(1024, activation='relu')(x5)
x2 = Dense(1024, activation='relu')(x6)
x3 = concatenate([x1, x2])
x3 = Dense(2048, activation='relu')(x3)
x3 = Dense(256, activation='relu')(x3)
pred = Dense(num_classes, activation='softmax', name='output')(x3)

model = Model(inputs=[inputs_xception, inputs_vgg], outputs=pred)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit({'xception_input': train_data_xception, 'vgg_input': train_data_vgg}, {'output': train_labels}, epochs=epochs, batch_size=batch_size, validation_data=({'xception_input': validation_data_xception, 'vgg_input': validation_data_vgg}, {'output': validation_labels}), class_weight=class_weights)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('siamese_loss.png')
plt.clf()


(eval_loss, eval_accuracy) = model.evaluate({'xception_input': validation_data_xception, 'vgg_input': validation_data_vgg}, {'output': validation_labels}, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))

valid_class_predicted = model.predict({'xception_input': validation_data_xception, 'vgg_input': validation_data_vgg})
cf = confusion_matrix(validation_labels2, np.argmax(valid_class_predicted, axis = 1))

fig, axis = plt.subplots()
heat_map = axis.pcolor(cf, cmap = plt.cm.Blues)
plt.colorbar(heat_map)
plt.savefig('siamese_cf.png')
plt.clf()

test_class_predicted = model.predict({'xception_input': test_data_xception, 'vgg_input': test_data_vgg})
test_class_predicted = np.argmax(test_class_predicted, axis=1)
test_file = np.load('test_files.npy')

class_dictionary = np.load('class_indices.npy').item()
inv_map = {v: k for k, v in class_dictionary.items()}

miss = np.zeros(12800)
sub_file = open('siamese_sub.csv', 'w')
sub_file.write("id,predicted\n")
for i in range(len(test_class_predicted)):
  sub_file.write("%d,%d\n" % (int(test_file[i][5:-4]), int(inv_map[test_class_predicted[i]])))
  miss[int(test_file[i][5:-4])-1] = 1

# There were less than 100 images that were missing due to bad/broken links
for i in range(12800):
  if miss[i] == 0:
    sub_file.write("%d,%d\n" % (i+1, 1))

sub_file.close()