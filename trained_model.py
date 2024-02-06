#!/usr/bin/python3

from ultralytics import YOLO
import os
import cv2
import torch


class Model:
    def __init__(self, weights_path_str):
        self.model = YOLO(weights_path_str)

    def yolo_detection(self, folder_path, results_path):
        torch.cuda.empty_cache()
        imgs = self.get_image_names(folder_path)
        for i, img in enumerate(imgs):
            res = self.model(img)[0].plot()
            cv2.imwrite(f"{results_path}/{i}.jpg", res)

    def yolo_finetune(self, path):
        torch.cuda.empty_cache()
        results = self.model.train(
            data=path,
            batch=12,
            epochs=1,
            save_json=True,
            pretrained=True
        )
        torch.save(results, 'best.pt')

    def get_image_names(self, folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует.")
            return None

        image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, file))
                       and any(file.lower().endswith(ext) for ext in image_extensions)]

        return image_paths