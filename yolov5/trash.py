import cv2
import torch
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import datetime
import sys
import time
import numpy as np

# Add YOLOv5 to path
sys.path.append('/home/2_fri/yolov5/yolov5/yolov5')

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

class TrashcanDetectorNode(Node):
    def __init__(self):
        super().__init__('trashcan_detector')
        self.bridge = CvBridge()
        
        # Directory for saving images
        self.send_dir = "send_images"
        os.makedirs(self.send_dir, exist_ok=True)
        self.clear_send_directory()  # Clear any existing images
        
        # Initialize camera
        self.setup_camera()
        
        # Initialize model if camera is working
        if self.direct_camera_mode:
            self.setup_model()
            
            # Create timer for processing frames
            self.create_timer(0.1, self.process_camera_frame)  # 10 Hz
            
            # Counter to track processed frames
            self.frame_count = 0
            
            # Variables to track detections and prevent duplicates
            self.last_detection_time = 0
            self.detection_cooldown = 3.0  # seconds between detections
            self.last_detection_bbox = None
            self.last_detection_conf = 0
            
            # Very high confidence threshold to eliminate false positives
            self.conf_threshold = 0.20  # Increased to 25% confidence
            
            self.iou_threshold = 0.5  # IOU threshold for considering a detection as duplicate
            
            # Flag to track if we're debugging (for debugging mode, set to True)
            self.debug_mode = False
            
            self.get_logger().info(f"Trashcan detector initialized with confidence threshold: {self.conf_threshold}")
        else:
            self.get_logger().error("Failed to find a working camera. Cannot initialize the detector.")
    
    def setup_camera(self):
        """Initialize and configure the camera"""
        self.cap = None
        self.direct_camera_mode = False
        
        # Get available camera devices
        available_cameras = [i for i in range(2) if os.path.exists(f"/dev/video{i}")]
        
        if not available_cameras:
            self.get_logger().error("No camera devices found")
            return
            
        self.get_logger().info(f"Found cameras: {available_cameras}")
        
        # Try to open each available camera
        for cam_idx in available_cameras:
            try:
                # Try opening with default settings
                cap = cv2.VideoCapture(cam_idx)
                
                if not cap.isOpened():
                    self.get_logger().warn(f"Could not open camera {cam_idx}")
                    continue
                
                # Configure settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                time.sleep(1)  # Give camera time to adjust
                
                # Read frame to verify
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    self.cap = cap
                    self.direct_camera_mode = True
                    self.get_logger().info(f"Successfully opened camera {cam_idx}")
                    
                    # Enable debug mode if having camera issues
                    # self.debug_mode = True
                    return
                else:
                    self.get_logger().warn(f"Camera {cam_idx} returned invalid frame")
                    cap.release()
                    
            except Exception as e:
                self.get_logger().error(f"Error with camera {cam_idx}: {str(e)}")
        
        # Try direct device path as fallback
        for cam_idx in available_cameras:
            try:
                cap = cv2.VideoCapture(f"/dev/video{cam_idx}")
                if cap.isOpened() and cap.read()[0]:
                    self.cap = cap
                    self.direct_camera_mode = True
                    self.get_logger().info(f"Opened camera using direct path: /dev/video{cam_idx}")
                    return
            except:
                pass
        
        self.get_logger().error("Failed to open any camera")
    
    def clear_send_directory(self):
        """Clear all files in the send_images directory"""
        try:
            # Delete all files but keep the directory
            for file in os.listdir(self.send_dir):
                file_path = os.path.join(self.send_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self.get_logger().info(f"Cleared all files from {self.send_dir}")
        except Exception as e:
            self.get_logger().error(f"Error clearing {self.send_dir}: {str(e)}")
    
    def setup_model(self):
        """Initialize the YOLOv5 model"""
        # Initialize the device
        self.device = select_device('')  # '' means CPU, or '0' for first GPU
        
        # Initialize the YOLOv5 model
        weights_path = '/home/2_fri/yolov5/yolov5/yolov5/runs/train/exp3/weights/best.pt'
        self.model = DetectMultiBackend(weights=weights_path, device=self.device)
        
        # Set stride and names
        self.stride = self.model.stride
        self.names = self.model.names
        self.get_logger().info(f"Model class names: {self.names}")
        self.get_logger().info(f"Number of classes: {len(self.names)}")
        
        # Set image size
        self.img_size = (640, 640)
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'direct_camera_mode') and self.direct_camera_mode and hasattr(self, 'cap') and self.cap:
            self.cap.release()
            cv2.destroyAllWindows()  # Close any open windows
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0  # No intersection
        
        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def is_duplicate_detection(self, bbox, conf):
        """Check if the current detection is a duplicate of the previous one"""
        # If no previous detection, this is not a duplicate
        if self.last_detection_bbox is None:
            return False
        
        # Check time since last detection
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            # Within cooldown period, check if it's the same object
            iou = self.calculate_iou(bbox, self.last_detection_bbox)
            
            # If high overlap and similar confidence, consider it a duplicate
            if iou > self.iou_threshold and abs(conf - self.last_detection_conf) < 0.05:
                return True
        
        # Not a duplicate
        return False
    
    def process_camera_frame(self):
        """Process frames from the camera"""
        if not self.direct_camera_mode or not self.cap:
            return
        
        try:
            # Refresh camera settings periodically
            if self.frame_count % 30 == 0:  # Every 30 frames
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
                self.cap.set(cv2.CAP_PROP_CONTRAST, 100)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                if self.debug_mode:
                    self.get_logger().info("Refreshed camera settings")
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                self.get_logger().warn("Failed to capture frame from camera")
                return
            
            self.frame_count += 1
            
            # Only log in debug mode
            if self.debug_mode:
                avg_pixel_value = np.mean(frame)
                self.get_logger().info(f"Processing frame #{self.frame_count} from direct camera (avg pixel value: {avg_pixel_value})")
                
                # Save raw frame periodically in debug mode
                if self.frame_count % 100 == 0:  # Every 100 frames
                    raw_filename = os.path.join(self.send_dir, f"raw_frame_{self.frame_count}.jpg")
                    cv2.imwrite(raw_filename, frame)
                    self.get_logger().info(f"Saved raw frame to {raw_filename}")
            
            # Process the frame
            self.detect_objects(frame)
            
        except Exception as e:
            self.get_logger().error(f"Error processing camera frame: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def detect_objects(self, frame):
        """Detect objects in the frame"""
        try:
            # Preprocess the image
            img = letterbox(frame, self.img_size, stride=self.stride)[0]
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # 0-255 to 0.0-1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            # Perform inference
            pred = self.model(img)
            
            # Apply NMS with very high confidence threshold
            pred = non_max_suppression(pred, 
                                      conf_thres=self.conf_threshold,  # Very high 60% confidence threshold
                                      iou_thres=0.45,  # IoU threshold
                                      classes=None)    # Filter by class
            
            # Process detections
            for i, det in enumerate(pred):  # per image
                # Only log number of detections in debug mode
                if self.debug_mode:
                    self.get_logger().info(f"Processing predictions: found {len(det)} objects")
                elif len(det) > 0:
                    # Only log when detections are found (not in debug mode)
                    self.get_logger().info(f"Found {len(det)} objects with confidence > {self.conf_threshold}")
                
                if len(det):
                    # Rescale boxes from img_size to frame size
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    # Draw boxes on frame
                    for *xyxy, conf, cls in det:
                        c = int(cls)
                        class_name = self.names[c] if c < len(self.names) else f"class_{c}"
                        
                        # Only process if it's a trashcan
                        if class_name.lower() == "trash_can":
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            current_bbox = [x1, y1, x2, y2]
                            
                            # Check if this is a duplicate detection
                            if not self.is_duplicate_detection(current_bbox, conf):
                                self.get_logger().info(f"Detected {class_name} with VERY HIGH confidence {conf:.2f}")
                                
                                # Draw bounding box on the image
                                label = f'{class_name} {conf:.2f}'
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Save detection image
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                filename = os.path.join(self.send_dir, f"detection_{class_name}_{timestamp}.jpg")
                                cv2.imwrite(filename, frame)
                                self.get_logger().info(f"Saved detection image to {filename}")
                                
                                # Update last detection information
                                self.last_detection_bbox = current_bbox
                                self.last_detection_conf = conf
                                self.last_detection_time = time.time()
                            else:
                                if self.debug_mode:
                                    self.get_logger().info(f"Skipped duplicate {class_name} detection")
                
        except Exception as e:
            self.get_logger().error(f"Error in detect_objects: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    
    node = TrashcanDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly by user")
    except Exception as e:
        node.get_logger().error(f"Error while spinning node: {str(e)}")
        import traceback
        node.get_logger().error(traceback.format_exc())
    finally:
        # Cleanup
        if hasattr(node, 'cap') and node.cap:
            node.cap.release()
        cv2.destroyAllWindows()  # Close any open windows
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
