import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import cv2, time, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)


class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, "r") as f:
            self.classesList = f.read().splitlines()

        # Colors list
        self.colorlist = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3)
        )

        print(len(self.classesList), len(self.colorlist))

    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[: fileName.index(".")]

        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(
            fname=fileName,
            origin=modelURL,
            cache_dir=self.cacheDir,
            cache_subdir="checkpoints",
            extract=True,
        )

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(
            os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model")
        )

        print("Model" + self.modelName + "loaded successfully")

    def createBoundingBox(self, image):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs,
            classScores,
            max_output_size=50,
            iou_threshold=0.5,
            score_threshold=0.5,
        )
        print(bboxIdx)

        if len(bboxs) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorlist[classIndex]

                displayText = f"{classLabelText}:{classConfidence}%"

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (
                    xmin * imW,
                    xmax * imW,
                    ymin * imH,
                    ymax * imH,
                )
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1
                )
            return image

    def predictImage(self, imagePath):
        image = cv2.imread(imagePath)

        bboxImage = self.createBoundingBox(image)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
