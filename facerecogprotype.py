import cv2
from deepface import DeepFace

known_face_path = r"C:\HACK\Facedatabse\know.jpg"

cap = cv2.VideoCapture(0)

print("Starting live face verification...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam. Exiting.")
        break


    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    try:
        faces = DeepFace.extract_faces(small_frame, enforce_detection=False)
    except Exception as e:
        print(f"Face detection error: {e}")
        faces = []

    verification_status = "No face detected"

    for face in faces:
        face_img = face["face"]
        box = face["facial_area"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]

        result = DeepFace.verify(face_img, known_face_path, enforce_detection=False)
        if result["verified"]:
            verification_status = "Verified"
        else:
            verification_status = "Unknown"


        x *= 2
        y *= 2
        w *= 2
        h *= 2

        color = (0, 255, 0) if verification_status == "Verified" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, verification_status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        break

    if not faces:
        cv2.putText(frame, verification_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Live Face Verification", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        print("Exiting loop by pressing 'q'")
        break

cap.release()
cv2.destroyAllWindows()
