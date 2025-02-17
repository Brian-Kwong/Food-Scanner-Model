import tensorflow as tf
import os
import matplotlib.pyplot as plt


#### Process the dataset and write images to disk

features = {
    "image/encoded": tf.io.FixedLenFeature(
        [], tf.string
    ),  # Image data in format specified by 'image/format' field (JPEG)
    "image/width": tf.io.FixedLenFeature(
        [], tf.int64
    ),  # Fixed length feature of width of the image
    "image/height": tf.io.FixedLenFeature(
        [], tf.int64
    ),  # Fixed length feature of height of the image
    "image/object/class/text": tf.io.VarLenFeature(
        tf.string
    ),  # Variable-length list of class names
    "image/object/class/label": tf.io.VarLenFeature(
        tf.int64
    ),  # Variable-length list of class labels
    "image/object/bbox/ymin": tf.io.VarLenFeature(
        tf.float32
    ),  # Variable-length list of bounding box ymin
    "image/object/bbox/ymax": tf.io.VarLenFeature(
        tf.float32
    ),  # Variable-length list of bounding box ymax
    "image/object/bbox/xmin": tf.io.VarLenFeature(
        tf.float32
    ),  # Variable-length list of bounding box xmin
    "image/object/bbox/xmax": tf.io.VarLenFeature(
        tf.float32
    ),  # Variable-length list of bounding box xmax
    "image/format": tf.io.FixedLenFeature([], tf.string),  # Image format (e.g., 'jpeg')
    "image/filename": tf.io.FixedLenFeature([], tf.string),  # Image filename
}


def model_parser(entry):
    entry = tf.io.parse_single_example(entry, features)
    image = tf.image.decode_jpeg(entry["image/encoded"], channels=3)

    # Collects all the features
    width = entry["image/width"]
    height = entry["image/height"]
    image_format = entry["image/format"]
    filename = entry["image/filename"]
    class_text = tf.sparse.to_dense(entry["image/object/class/text"])
    class_label = tf.sparse.to_dense(entry["image/object/class/label"])
    ymin = tf.sparse.to_dense(entry["image/object/bbox/ymin"])
    ymax = tf.sparse.to_dense(entry["image/object/bbox/ymax"])
    xmin = tf.sparse.to_dense(entry["image/object/bbox/xmin"])
    xmax = tf.sparse.to_dense(entry["image/object/bbox/xmax"])
    return (
        image,
        {
            "width": width,
            "height": height,
            "image_format": image_format,
            "filename": filename,
            "class_text": class_text,
            "class_label": class_label,
            "ymin": ymin,
            "ymax": ymax,
            "xmin": xmin,
            "xmax": xmax,
        },
    )


def write_images(dataset, dataset_type):
    BASE_PATH = os.path.join(os.getcwd(), "images", dataset_type)
    if os.path.exists(BASE_PATH):
        print("Images already written for ", dataset_type, "Skipping...")
        return
    os.makedirs(BASE_PATH, exist_ok=True)
    i = 0
    for image, features in dataset:
        filename = features["filename"].numpy().decode("utf-8")

        image = image.numpy()

        height = features["height"].numpy()
        width = features["width"].numpy()
        categories = features["class_text"].numpy()
        for j, category in enumerate(categories):

            category_path = os.path.join(BASE_PATH, category.decode("utf-8"))
            if not os.path.exists(category_path):
                os.makedirs(category_path, exist_ok=True)

            ymin = features["ymin"].numpy()[j]
            ymax = features["ymax"].numpy()[j]
            xmin = features["xmin"].numpy()[j]
            xmax = features["xmax"].numpy()[j]
            image_cropped = image[
                int(ymin * height) - 30 : int(ymax * height) + 30,
                int(xmin * width) - 30 : int(xmax * width) + 30,
            ]
            if image_cropped.size == 0:
                continue
            plt.imsave(f"{category_path}/{filename}.jpg", image_cropped)
            i += 1
    print("Done writing images for ", dataset_type)


TF_TRAIN_RECORDS = "./datasetV3/train/food.tfrecord"
TF_VALIDATION_RECORDS = "./datasetV3/valid/food.tfrecord"
TF_TEST_RECORDS = "./datasetV3/test/food.tfrecord"
train_dataset = tf.data.TFRecordDataset(TF_TRAIN_RECORDS)
train_dataset = train_dataset.map(model_parser)
validation_dataset = tf.data.TFRecordDataset(TF_VALIDATION_RECORDS)
validation_dataset = validation_dataset.map(model_parser)
test_dataset = tf.data.TFRecordDataset(TF_TEST_RECORDS)
test_dataset = test_dataset.map(model_parser)

write_images(train_dataset, "train")
write_images(validation_dataset, "validation")
write_images(test_dataset, "test")

#######

# Use Image Classification model from TensorLite Model Maker
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader
from tflite_model_maker.config import QuantizationConfig


train_data = DataLoader.from_folder("images/train")
validation_data = DataLoader.from_folder("images/validation")
test_data = DataLoader.from_folder("images/test")

os.makedirs("model", exist_ok=True)


model = image_classifier.create(
    train_data,
    model_spec=image_classifier.EfficientNetLite4Spec(),
    validation_data=validation_data,
    epochs=20,
)
loss, accuracy = model.evaluate(test_data)
config = QuantizationConfig.for_float16()
model.export(
    export_dir="model", tflite_filename="model.tflite", quantization_config=config
)
print("Loss: ", loss, "Accuracy: ", accuracy)
print("Model exported to ./model directory")
