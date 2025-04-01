import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import torch
from torchvision import models, transforms

class ObjectRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Recognition App")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.root.destroy()
            return
        
        # Load pre-trained model
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        
        # Load ImageNet class labels
        with open('imagenet_classes.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        # Create custom categories dictionary
        self.custom_categories = {
            'banana': {'info': 'Edible - Fruit', 'waste': 'Compost'},
            'key': {'info': 'Not edible', 'waste': 'Recycling (metal)'},
            'toothbrush': {'info': 'Not edible', 'waste': 'Plastic waste'},
            'apple': {'info': 'Edible - Fruit', 'waste': 'Compost'},
            'orange': {'info': 'Edible - Fruit', 'waste': 'Compost'},
            'spoon': {'info': 'Not edible', 'waste': 'Metal recycling'},
            'fork': {'info': 'Not edible', 'waste': 'Metal recycling'},
            'knife': {'info': 'Not edible', 'waste': 'Metal recycling'},
            'bottle': {'info': 'Not edible', 'waste': 'Plastic recycling'},
            'cup': {'info': 'Not edible', 'waste': 'Check material'},
        }
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create GUI
        self.create_widgets()
        
        # Start video feed
        self.show_frame()
    
    def create_widgets(self):
        # Video frame
        self.video_frame = ttk.Label(self.root)
        self.video_frame.pack(padx=10, pady=10)
        
        # Capture button
        self.capture_btn = ttk.Button(self.root, text="Capture & Recognize", command=self.capture_image)
        self.capture_btn.pack(pady=5)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(self.root, text="Recognition Result", padding=10)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Object name label
        self.object_name = ttk.Label(self.result_frame, text="Object: ", font=('Helvetica', 12, 'bold'))
        self.object_name.pack(anchor=tk.W)
        
        # Edible info label
        self.edible_info = ttk.Label(self.result_frame, text="Edible: ", font=('Helvetica', 10))
        self.edible_info.pack(anchor=tk.W)
        
        # Waste info label
        self.waste_info = ttk.Label(self.result_frame, text="Waste Category: ", font=('Helvetica', 10))
        self.waste_info.pack(anchor=tk.W)
        
        # Exit button
        self.exit_btn = ttk.Button(self.root, text="Exit", command=self.close_app)
        self.exit_btn.pack(pady=10)
    
    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to RGB and resize for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            
            # Convert to ImageTk format
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            
        # Repeat every 10ms
        self.root.after(10, self.show_frame)
    
    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert to PIL Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Preprocess and predict
            input_tensor = self.transform(img)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_batch)
            
            # Get prediction
            _, index = torch.max(output, 1)
            percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
            predicted_class = self.classes[index[0]]
            
            # Extract the simple object name (remove n###### prefix)
            simple_name = predicted_class.split(',')[0].split(' ')[0].lower()
            
            # Update GUI with results
            self.object_name.config(text=f"Object: {simple_name.capitalize()}")
            
            # Get custom info if available
            edible_text = "Unknown"
            waste_text = "Unknown"
            
            for obj, info in self.custom_categories.items():
                if obj in simple_name:
                    edible_text = info['info']
                    waste_text = info['waste']
                    break
            
            self.edible_info.config(text=f"Edible: {edible_text}")
            self.waste_info.config(text=f"Waste Category: {waste_text}")
    
    def close_app(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    # Download ImageNet classes file if not exists
    import urllib.request
    import os
    
    if not os.path.exists('imagenet_classes.txt'):
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.txt"
        urllib.request.urlretrieve(url, 'imagenet_classes.txt')
    
    root = tk.Tk()
    app = ObjectRecognitionApp(root)
    root.mainloop()