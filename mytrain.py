from keras.applications import MobileNet
from keras.models import Model
from keras.layers import (
    Dense, Dropout, Activation, Flatten, 
    GlobalAveragePooling2D, BatchNormalization
)
from keras.preprocessing.image import ImageDataGenerator

# Input image dimensions for MobileNet
IMG_ROWS, IMG_COLS = 224, 224

# Load MobileNet with pre-trained weights and exclude the top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(IMG_ROWS, IMG_COLS, 3))

# Unfreeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = True

# Print the trainable status of each layer
for idx, layer in enumerate(base_model.layers):
    print(f"Layer {idx}: {layer.__class__.__name__}, Trainable: {layer.trainable}")

def create_top_model(base, num_classes):
    """
    Builds the top layers for the MobileNet model.
    """
    top = base.output
    top = GlobalAveragePooling2D()(top)
    top = Dense(1024, activation='relu')(top)
    top = Dense(1024, activation='relu')(top)
    top = Dense(512, activation='relu')(top)
    top = Dense(num_classes, activation='softmax')(top)
    return top

# Number of emotion classes
NUM_CLASSES = 5

# Create the full model
top_layers = create_top_model(base_model, NUM_CLASSES)
model = Model(inputs=base_model.input, outputs=top_layers)

# Display model summary
print(model.summary())

# Directories for training and validation datasets
TRAIN_DIR = '/mycomputer/Deep_Learning/Emotion_Classification/fer2013/train'
VALIDATION_DIR = '/mycomputer/Deep_Learning/Emotion_Classification/fer2013/validation'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Batch size for training and validation
BATCH_SIZE = 32

# Generate batches of training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Generate batches of validation data
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_ROWS, IMG_COLS),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Callbacks for training
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Save the best model during training
checkpoint = ModelCheckpoint(
    'emotion_face_mobilNet.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Stop training early if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate on plateau
lr_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', 
    factor=0.2, 
    patience=5, 
    min_lr=0.0001, 
    verbose=1
)

# List of callbacks
callbacks = [checkpoint, early_stop, lr_reduction]

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Dataset parameters
NUM_TRAIN_SAMPLES = 24176
NUM_VALIDATION_SAMPLES = 3006
EPOCHS = 25

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=NUM_TRAIN_SAMPLES // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=NUM_VALIDATION_SAMPLES // BATCH_SIZE
)
