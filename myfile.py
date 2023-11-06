# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load the competition dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Data Exploration
print("Shape of the training data:", train_df.shape)
print("Shape of the test data:", test_df.shape)
print("Sample data:")
print(train_df.head())

# Class Distribution
print("Number of samples in each class:\n", train_df['target'].value_counts())
sns.countplot(x='target', data=train_df)
plt.show()

# Preprocessing and Data Augmentation
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32

# Define data generators for training and validation
datagen = ImageDataGenerator(rescale=1./255,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest',
                            validation_split=0.2)

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                              directory="train_images/",
                                              x_col="id",
                                              y_col="target",
                                              target_size=IMAGE_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode="binary",
                                              subset="training")

validation_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                   directory="train_images/",
                                                   x_col="id",
                                                   y_col="target",
                                                   target_size=IMAGE_SIZE,
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="binary",
                                                   subset="validation")

# Model Building and Training
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=train_generator.n//BATCH_SIZE,
                    validation_data=validation_generator, validation_steps=validation_generator.n//BATCH_SIZE,
                    epochs=10)

# Model Evaluation
y_pred = model.predict(validation_generator)
y_true = validation_generator.classes

roc_auc = roc_auc_score(y_true, y_pred)
print("ROC AUC:", roc_auc)

classification_rep = classification_report(y_true, (y_pred > 0.5).astype(int))
print("Classification Report:\n", classification_rep)

# Visualization of training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Discussion and Conclusion
# Include your discussion and conclusions here.
