import cv2
import torch
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TrashcanDetectorNode(Node):
    def __init__(self):
        super().__init__('trashcan_detector')
        self.bridge = CvBridge()
        
        # Initialize the YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/trashcan_detector/weights/best.pt')
        # Set up a subscriber to the camera topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Replace with the actual camera topic
            self.image_callback,
            10
        )

        # Directory where detected images will be saved
        self.send_dir = "send_images"
        os.makedirs(self.send_dir, exist_ok=True)

    def image_callback(self, msg):
        # Convert the ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Perform inference on the frame
        results = self.model(frame)

        # Render the results (bounding boxes and labels)
        results.render()

        # Loop through the results and check for trashcan detections (assuming class 0 is 'trashcan')
        for *xywh, conf, cls in results.xywh[0]:
            if int(cls) == 0:
                self.get_logger().info("Trashcan detected!")

                # Generate a unique filename using the current time
                filename = os.path.join(self.send_dir, f"trashcan_{cv2.getTickCount()}.jpg")
                
                # Save the current frame to the send_images folder
                cv2.imwrite(filename, frame)

def main(args=None):
    rclpy.init(args=args)
    
    node = TrashcanDetectorNode()
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()