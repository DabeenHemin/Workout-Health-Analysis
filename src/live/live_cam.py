#Import Packages

import cv2 # OpenCVis an open-source software toolkit for computer vision and machine learning tasks. In this case the cv2 libray allows for image reading and processing.
import mediapipe as mp #The MediaPipe Pose Landmarker task lets us detect up to 33 landmarks of individuals body in an image or video. This will be used to identify key body locations, analyze posture, and categorize movements.

mp_pose = mp.solutions.pose #This gives access to the pose estimation model which in this will inspect the live cam footage, detect if there is a human body in its presence and then return 33 landmarks of a user (key points such as shoulders, hips, knees ect.)
mp_drawing = mp.solutions.drawing_utils #This allows the system to gives us a visualisation of how a users body is moving by taking the landmarks of them and drawing connections between them which then creates a skeleton overlay.

live_cam =cv2.VideoCapture(0)

if not live_cam.isOpened():
    print("Live cam not enabled")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: #Here is where the pose detector is implemented and created. The 2 cofindece values tells the pose detector how confident it has to be to dectect a person and how confident it is to detect a pose and keep following it smoothly across frames
    while live_cam.isOpened():
        ret, frame = live_cam.read()

        if not ret:
            print("Live feed over")
            break

        frame = cv2.flip(frame, 1) # Flips the live cam to make it mirror like and not confuse a user whilst using it

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=6),
            )

        cv2.imshow("Live Webcam",image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

live_cam.release()
cv2.destroyAllWindows()

