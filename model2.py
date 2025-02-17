import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json

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
    BASE_PATH = os.path.join(os.getcwd(), "images")
    os.makedirs(BASE_PATH, exist_ok=True)
    csv_array = []
    i = 0
    for image, features in dataset:
        filename = "IMG_" + str(i) + ".jpg"
        image = image.numpy()
        categories = features["class_text"].numpy()
        for j, category in enumerate(categories):

            category_path = category.decode("utf-8")

            ymin = features["ymin"].numpy()[j]
            ymax = features["ymax"].numpy()[j]
            xmin = features["xmin"].numpy()[j]
            xmax = features["xmax"].numpy()[j]
            plt.imsave(f"{BASE_PATH}/{filename}", image)
            csv_array.append(
                f"{dataset_type},{filename},{category_path},{xmin},{ymin},{xmax},{ymin},{xmax},{ymax},{xmin},{ymax}"
            )
        i += 1
    print("Done writing images for ", dataset_type)
    return csv_array


TF_TRAIN_RECORDS = "./datasetV3/train/food.tfrecord"
TF_VALIDATION_RECORDS = "./datasetV3/valid/food.tfrecord"
TF_TEST_RECORDS = "./datasetV3/test/food.tfrecord"
train_dataset = tf.data.TFRecordDataset(TF_TRAIN_RECORDS)
train_dataset = train_dataset.map(model_parser)
validation_dataset = tf.data.TFRecordDataset(TF_VALIDATION_RECORDS)
validation_dataset = validation_dataset.map(model_parser)
test_dataset = tf.data.TFRecordDataset(TF_TEST_RECORDS)
test_dataset = test_dataset.map(model_parser)

if not os.path.exists("images"):
    csv_array = []
    csv_array += write_images(train_dataset, "TRAIN")
    csv_array += write_images(validation_dataset, "VALIDATE")
    csv_array += write_images(test_dataset, "TEST")
    with open("./images/images.csv", "w") as f:
        for item in csv_array:
            f.write("%s\n" % item)

####################################################################################################
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import numpy as np


def seralizeModel(model_output):
    if isinstance(model_output, dict):
        return {k: seralizeModel(v) for k, v in model_output.items()}
    elif isinstance(model_output, list):
        return [seralizeModel(v) for v in model_output]
    elif isinstance(model_output, np.float32):
        return float(model_output)
    elif isinstance(model_output, np.int32):
        return int(model_output)
    else:
        return model_output


os.chdir("./images")
spec = model_spec.get("efficientdet_lite0")
train_data, validation_data, test_data = object_detector.DataLoader.from_csv(
    "images.csv"
)
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=8,
    epochs=10,
    train_whole_model=True,
    validation_data=validation_data,
)
model.summary()
output = model.evaluate(test_data)
output = seralizeModel(output)
os.chdir("..")
with open("output.json", "w") as f:
    json.dump(output, f)

config = QuantizationConfig.for_float16()
model.export(
    export_dir="./model",
    tflite_filename="model.tflite",
    quantization_config=config,
)
