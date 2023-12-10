from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load the MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


# Replace with the path of your default necklace image
default_necklace_image_path = 'static/Image/Necklace/necklace_1.png'


# Create a VideoCapture object to capture video from the webcam (index 0)
cap = cv2.VideoCapture(0)

def generate_frames(necklace_image_path):

    # Load the necklace image
    necklace_image = cv2.imread(necklace_image_path, cv2.IMREAD_UNCHANGED)

    while True:
        # Read the current frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get the width and height of the frame
        height, width, _ = frame.shape

        # Calculate the width of each section
        center_width = int(width * 0.35)
        side_width = (width - center_width) // 2

        # Divide the frame into three sections
        left_section = frame[:, :side_width]
        center_section = frame[:, side_width:side_width + center_width]
        right_section = frame[:, side_width + center_width:]

        # Draw vertical lines to split the screen
        #cv2.line(frame, (side_width, 0), (side_width, height), (0, 255, 0), 1)
        #cv2.line(frame, (side_width + center_width, 0), (side_width + center_width, height), (0, 255, 0), 1)

        # Process the center section for face detection and overlay
        # Process the center section for face detection and overlay
        # Process the center section for face detection and overlay
        frame_rgb = cv2.cvtColor(center_section, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
            
        if results.detections:
            for detection in results.detections:
                # Iterate over the landmarks and draw them on the frame
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    # Get the pixel coordinates of the landmark
                    cx, cy = int(landmark.x * center_width), int(landmark.y * height)

                    # Check if hand is within the valid region
                    if cx >= 0 and cx <= center_width:
                        hand_in_frame = True

                        # Extract the bounding box coordinates
                        bboxC = detection.location_data.relative_bounding_box
                        hC, wC, _ = center_section.shape
                        xminC = int(bboxC.xmin * wC)
                        yminC = int(bboxC.ymin * hC)
                        widthC = int(bboxC.width * wC)
                        heightC = int(bboxC.height * hC)
                        xmaxC = xminC + widthC
                        ymaxC = yminC + heightC

                        # Calculate the bottom bounding box coordinates
                        bottom_ymin = ymaxC + 10
                        bottom_ymax = min(ymaxC + 150, hC)

                        # Increase the width of the red bounding box
                        xminC -= 20  # Decrease the left side
                        xmaxC += 20  # Increase the right side

                        # Check if the bounding box dimensions are valid
                        if widthC > 0 and heightC > 0 and xmaxC > xminC and bottom_ymax > bottom_ymin:
                            # Resize necklace image to fit the bounding box size
                            resized_image = cv2.resize(necklace_image, (xmaxC - xminC, bottom_ymax - bottom_ymin))

                            # Calculate the start and end coordinates for the necklace image
                            start_x = xminC
                            start_y = bottom_ymin
                            end_x = start_x + (xmaxC - xminC)
                            end_y = start_y + (bottom_ymax - bottom_ymin)

                            # Create a mask from the alpha channel
                            alpha_channel = resized_image[:, :, 3]
                            mask = alpha_channel[:, :, np.newaxis] / 255.0

                            # Apply the mask to the necklace image
                            overlay = resized_image[:, :, :3] * mask

                            # Create a mask for the input image region
                            mask_inv = 1 - mask

                            # Apply the inverse mask to the input image
                            region = center_section[start_y:end_y, start_x:end_x]
                            resized_mask_inv = None
                            if region.shape[1] > 0 and region.shape[0] > 0:
                                resized_mask_inv = cv2.resize(mask_inv, (region.shape[1], region.shape[0]))
                                resized_mask_inv = resized_mask_inv[:, :, np.newaxis]  # Add an extra dimension to match the number of channels

                            if resized_mask_inv is not None:
                                region_inv = region * resized_mask_inv

                                # Combine the resized image and the input image region
                                resized_overlay = None
                                if region_inv.shape[1] > 0 and region_inv.shape[0] > 0:
                                    resized_overlay = cv2.resize(overlay, (region_inv.shape[1], region_inv.shape[0]))

                                # Combine the resized overlay and region_inv
                                region_combined = cv2.add(resized_overlay, region_inv)

                                # Replace the neck region in the input image with the combined region
                                center_section[start_y:end_y, start_x:end_x] = region_combined    

        # Concatenate the sections back into a single frame
        frame = np.concatenate((left_section, center_section, right_section), axis=1)

        # Display the frame
        if not hand_in_frame:
            cv2.putText(frame, "You are not in the frame. Come closer to the frame.", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Reset the flag for the next frame
        hand_in_frame = False

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as a byte string
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')    
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    design = request.args.get('selected_accessory')
  

    necklace_image_path = request.args.get('necklace_image_path', default=default_necklace_image_path)
    return Response(generate_frames(necklace_image_path), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host= "0.0.0.0",port= 8080, debug=True)
