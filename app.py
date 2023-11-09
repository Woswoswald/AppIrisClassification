import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import pickle

with open('label_encoderrrr.pkl', 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

def preprocess_img(img, new_dim=(240, 320)):
    new_img = cv2.resize(img, (new_dim[1], new_dim[0]), interpolation=cv2.INTER_AREA)
    mean = np.mean(new_img)
    std = np.std(new_img)
    new_img = (new_img - mean) / std
    return new_img

class ImagePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application d'Authentification")

        self.load_model()  
        

        self.select_button = tk.Button(root, text="Sélectionner une image", command=self.load_image)
        self.select_button.pack()
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        self.predict_button = tk.Button(root, text="Lancer la Prédiction", command=self.predict_image)
        self.predict_button.pack()

        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack()

    def load_model(self):
        # model_path = "C:/Users/utilisateur/Documents/vscode_env/cas_pratique/CAS_PRATIQUES/code/lr_classifier_v1.keras"
        self.model = tf.keras.models.load_model("lr_classifier2_v1.keras")
        self.left_model = tf.keras.models.load_model("left_model.keras")
        self.right_model = tf.keras.models.load_model("right_model.keras")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image_prep = preprocess_img(self.image)
            self.imagepath_label.config(text=f"Image : {file_path}")
            self.photo = ImageTk.PhotoImage(Image.open(file_path))
            self.imagedisp_label.config(image=self.photo)

    def predict_image(self):
        if hasattr(self, 'image') and self.model is not None:
            prediction = self.model.predict(np.array([self.image_prep]))
            predicted_class = np.argmax(prediction)
            proba = prediction.max()
            class_labels = ["Left Eye", "Right Eye"]
            predicted_label = class_labels[predicted_class]

            initial_prediction_text = f"Prédiction du modèle initial : {predicted_label}, {proba}"

            if predicted_label == "Right Eye":
                if hasattr(self, 'right_model'):
                    right_prediction = self.right_model.predict(np.array([self.image_prep]))
                    right_predicted_class = np.argmax(right_prediction)
                    right_proba = right_prediction.max()
                    decoded_right_label = label_encoder.inverse_transform([right_predicted_class])[0]
                    right_prediction_text = f"Prédiction du modèle droit : {decoded_right_label}, {right_proba}"
                    self.prediction_label.config(text=f"{initial_prediction_text}\n{right_prediction_text}")


            elif predicted_label == "Left Eye":
                if hasattr(self, 'left_model'):
                    left_prediction = self.left_model.predict(np.array([self.image_prep]))
                    left_predicted_class = np.argmax(left_prediction)
                    left_proba = left_prediction.max()
                    decoded_left_label = label_encoder.inverse_transform([left_predicted_class])[0]
                    left_prediction_text = f"Prédiction du modèle gauche : {decoded_left_label}, {left_proba}"
                    self.prediction_label.config(text=f"{initial_prediction_text}\n{left_prediction_text}")
                    

            else:
                self.prediction_label.config(text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictionApp(root)
    root.mainloop()

        #     self.prediction_label.config(text=f"Prédiction du modèle : {predicted_label}, {proba}")
        # else:
        #     self.prediction_label.config(text="Aucune image sélectionnée ou modèle non chargé")

