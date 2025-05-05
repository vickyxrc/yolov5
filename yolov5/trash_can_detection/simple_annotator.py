import os
import glob
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class SimpleAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple YOLO Annotator for Trash Cans")
        
        # Paths
        self.image_dir = "data/images/train"
        self.label_dir = "data/labels/train"
        
        # Ensure label directory exists
        os.makedirs(self.label_dir, exist_ok=True)
        
        # Get all webp files
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.webp")))
        self.current_image_index = 0
        
        # Main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display and annotation
        self.canvas = tk.Canvas(self.main_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        self.control_frame = ttk.Frame(root, padding="5")
        self.control_frame.pack(fill=tk.X)
        
        # Navigation buttons
        self.prev_button = ttk.Button(self.control_frame, text="Previous (A)", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.control_frame, text="Next (D)", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        # Status display
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self.control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Setup rectangle drawing
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Key bindings
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        # Load the first image
        self.load_image()
    
    def load_image(self):
        if not self.image_files:
            self.status_var.set("No images found!")
            return
        
        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)
        self.status_var.set(f"Image {self.current_image_index + 1}/{len(self.image_files)}: {image_name}")
        
        try:
            # Open and display image
            img = Image.open(image_path)
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_width, self.image_height = img.size
            
            # Configure canvas size
            self.canvas.config(width=min(self.image_width, 800), height=min(self.image_height, 600))
            
            # Clear canvas and display new image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Load existing annotations
            self.load_annotations(image_path)
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {e}")
    
    def load_annotations(self, image_path):
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # Convert normalized coordinates to pixel values
                        x1 = int((x_center - width/2) * self.image_width)
                        y1 = int((y_center - height/2) * self.image_height)
                        x2 = int((x_center + width/2) * self.image_width)
                        y2 = int((y_center + height/2) * self.image_height)
                        
                        # Draw rectangle
                        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="box")
    
    def on_press(self, event):
        # Store starting position
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        # Create new rectangle
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2, tags="temp_rect"
        )
    
    def on_drag(self, event):
        # Update rectangle as mouse moves
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
    
    def on_release(self, event):
        # Finalize rectangle on mouse release
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Only create annotation if the box has meaningful size
        if abs(end_x - self.start_x) > 5 and abs(end_y - self.start_y) > 5:
            # Delete temporary rectangle
            self.canvas.delete("temp_rect")
            
            # Create permanent rectangle
            self.canvas.create_rectangle(
                self.start_x, self.start_y, end_x, end_y,
                outline="red", width=2, tags="box"
            )
            
            # Save annotation
            self.save_annotation(self.start_x, self.start_y, end_x, end_y)
        else:
            # Delete temporary rectangle if too small
            self.canvas.delete("temp_rect")
        
        self.rect = None
    
    def save_annotation(self, x1, y1, x2, y2):
        # Get image file path
        image_path = self.image_files[self.current_image_index]
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")
        
        # Convert to YOLO format (normalized)
        x_min = min(x1, x2) / self.image_width
        y_min = min(y1, y2) / self.image_height
        x_max = max(x1, x2) / self.image_width
        y_max = max(y1, y2) / self.image_height
        
        # Calculate center and dimensions (YOLO format)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Generate YOLO format line (class_id x_center y_center width height)
        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        # Append to annotation file
        with open(label_path, 'a') as f:
            f.write(yolo_line)
        
        self.status_var.set(f"Added annotation to {base_name}.txt")
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleAnnotator(root)
    root.mainloop()
