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



Test evaluation:  {'AP': 0.0016256366, 'AP50': 0.0036999339, 'AP75': 0.0012053173, 'APs': 0.0, 'APm': 2.6319305e-05, 'APl': 0.0016747784, 'ARmax1': 0.039136704, 'ARmax10': 0.057153244, 'ARmax100': 0.057330295, 'ARs': 0.0, 'ARm': 0.002963703, 'ARl': 0.059264433, 'AP_/Bethu ko Saag': 0.0, 'AP_/Bitter Gourd': 0.00081004796, 'AP_/Mushroom': 0.0035297312, 'AP_/Sweet Potato -Suthuni-': 0.0013185419, 'AP_/Beans': 0.0029435975, 'AP_/mayonnaise': 0.0, 'AP_/Bread': 0.0, 'AP_/Egg': 0.0027542596, 'AP_/Taro Leaves -Karkalo-': 7.4316384e-05, 'AP_/Chili Pepper -Khursani-': 0.0051445523, 'AP_/Chickpeas': 0.00044179865, 'AP_/Red Lentils': 0.050896797, 'AP_/Cabbage': 0.00097401114, 'AP_/Sajjyun -Moringa Drumsticks-': 0.0, 'AP_/Beetroot': 0.0005414961, 'AP_/Cinnamon': 0.0018756734, 'AP_/Stinging Nettle -Sisnu-': 0.0006186667, 'AP_/Artichoke': 6.970841e-05, 'AP_/Lemon -Nimbu-': 0.0, 'AP_/Buff Meat': 4.4078977e-05, 'AP_/Palungo -Nepali Spinach-': 0.010822418, 'AP_/Black Lentils': 0.0037940461, 'AP_/Fiddlehead Ferns -Niguro-': 6.900247e-05, 'AP_/noodle': 0.0, 'AP_/Onion Leaves': 0.0, 'AP_/Carrot': 3.7588507e-05, 'AP_/Onion': 0.0, 'AP_/Tomato': 0.00032354586, 'AP_/Chayote-iskus-': 9.340557e-06, 'AP_/Asparagus -Kurilo-': 0.0004863372, 'AP_/Garlic': 1.2779951e-05, 'AP_/Ginger': 0.0, 'AP_/Chili Powder': 0.0, 'AP_/Sponge Gourd -Ghiraula-': 0.0020199076, 'AP_/Avocado': 9.5750096e-05, 'AP_/seaweed': 0.0, 'AP_/Nutrela -Soya Chunks-': 0.0, 'AP_/Pointed Gourd -Chuche Karela-': 0.00014368711, 'AP_/Butter': 0.0, 'AP_/Cheese': 0.0, 'AP_/Thukpa Noodles': 0.0, 'AP_/Broad Beans -Bakullo-': 0.0, 'AP_/Brinjal': 6.291726e-06, 'AP_/Capsicum': 0.00012550311, 'AP_/Gundruk': 0.0042655915, 'AP_/Cucumber': 2.0954476e-06, 'AP_/Beef': 0.0, 'AP_/Tree Tomato -Rukh Tamatar-': 0.00017472336, 'AP_/Ham': 0.0, 'AP_/Garden cress-Chamsur ko saag-': 0.007928832, 'AP_/Jack Fruit': 0.00077959197, 'AP_/Rahar ko Daal': 0.0, 'AP_/Potato': 0.0029432178, 'AP_/Soy Sauce': 0.0, 'AP_/Coriander -Dhaniya-': 0.01953406, 'AP_/Green Brinjal': 0.0, 'AP_/Rice -Chamal-': 0.0064511076, 'AP_/Milk': 0.0, 'AP_/kimchi': 0.0, 'AP_/Corn': 0.0, 'AP_/Pumpkin -Farsi-': 0.0, 'AP_/Okra -Bhindi-': 0.0020822173, 'AP_/Farsi ko Munta': 0.0003793872, 'AP_/Strawberry': 0.0, 'AP_/Bacon': 0.0, 'AP_/Paneer': 1.826751e-05, 'AP_/Pork': 0.0, 'AP_/Papaya': 0.000102444625, 'AP_/Salt': 0.0, 'AP_/Green Soyabean -Hariyo Bhatmas-': 0.0, 'AP_/Crab Meat': 0.0, 'AP_/Taro Root-Pidalu-': 0.0, 'AP_/Lapsi -Nepali Hog Plum-': 1.8578885e-05, 'AP_/Radish': 3.5577745e-05, 'AP_/Palak -Indian Spinach-': 0.0, 'AP_/Garden Peas': 4.2253396e-05, 'AP_/Ash Gourd -Kubhindo-': 0.0014749935, 'AP_/Snake Gourd -Chichindo-': 1.0870448e-05, 'AP_/Banana': 0.00023296448, 'AP_/Masyaura': 5.531079e-05, 'AP_/Soyabean -Bhatmas-': 0.00037749167, 'AP_/Chicken': 0.0, 'AP_/Black beans': 0.00924543, 'AP_/Moringa Leaves -Sajyun ko Munta-': 0.0, 'AP_/Mutton': 0.0006135969, 'AP_/Cauliflower': 0.0005424349, 'AP_/Broccoli': 1.3847539e-05, 'AP_/Turnip': 0.010681661, 'AP_/Bottle Gourd -Lauka-': 0.0021463786, 'AP_/Green Mint -Pudina-': 0.0006336376, 'AP_/Rayo ko Saag': 0.00198069, 'AP_/Green Lentils': 0.015440125, 'AP_/Cassava -Ghar Tarul-': 0.00019724778, 'AP_/Cornflakec': 0.0, 'AP_/Bamboo Shoots -Tama-': 0.00017074497, 'AP_/Sausage': 0.0, 'AP_/Ketchup': 0.0, 'AP_/Tori ko Saag': 0.0, 'AP_/Lime -Kagati-': 0.00048729384, 'AP_/Chicken Gizzards': 0.0, 'AP_/Beaten Rice -Chiura-': 0.0, 'AP_/Minced Meat': 7.6660435e-05, 'AP_/Red Beans': 0.007825406, 'AP_/Sugar': 0.0, 'AP_/Tofu': 0.0, 'AP_/Ice': 0.0, 'AP_/Green Peas': 0.0, 'AP_/Olive Oil': 0.0, 'AP_/Akabare Khursani': 0.0, 'AP_/Long Beans -Bodi-': 0.0, 'AP_/Chowmein Noodles': 0.0, 'AP_/Water Melon': 0.0, 'AP_/Wallnut': 0.0, 'AP_/Yellow Lentils': 0.0, 'AP_/Orange': 0.0, 'AP_/Fish': -1.0, 'AP_/Wheat': -1.0, 'AP_/Pea': -1.0, 'AP_/Apple': -1.0, 'AP_/Pear': -1.0}