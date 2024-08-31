#Import the required library 

from ultralytics import YOLO
import torch

def trainModelonCustomDataset():
    #Select the Yolo model

    model = YOLO("yolov8n.pt")

    #train the model on the required dataset
    model.train(data = "/content/safety-Helmet-Reflective-Jacket/data.yaml", epochs = 20)


if __name__ == "__main__":
    trainModelonCustomDataset()