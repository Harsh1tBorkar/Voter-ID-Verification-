from DB.db import VoterDatabase
import cv2
from deepface import DeepFace
import os
import sys

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


try:
    db = VoterDatabase()
except Exception as e:
    print(f"Error initializing database: {e}")
    sys.exit(1)

def verify_live_face(known_face_path):
    if not os.path.exists(known_face_path):
        print(f"Known face image not found at {known_face_path}")
        return False

    cap = cv2.VideoCapture(0)
    print("Starting live face verification. Press Ctrl+C to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
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
                try:
                    result = DeepFace.verify(face_img, known_face_path, enforce_detection=False)
                    verification_status = "Verified" if result["verified"] else "Unknown"
                except Exception as e:
                    print(f"Verification error: {e}")
                    verification_status = "Error"

                x, y, w, h = face["facial_area"].values()
                x, y, w, h = x*2, y*2, w*2, h*2
                color = (0, 255, 0) if verification_status == "Verified" else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, verification_status, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if verification_status == "Verified":
                    print("Face verification successful!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                break

            if not faces:
                cv2.putText(frame, verification_status, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow("Live Face Verification", frame)
            cv2.waitKey(10)

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return False

def main():
    voter_id = input("Enter Voter ID: ").strip()
    aadhaar = input("Enter Aadhaar number: ").strip()

    record = db.find_voter(voter_id)
    if record is None:
        print("Voter ID not found.")
        return

    if record.get("aadhaar_number") != aadhaar:
        print("Aadhaar number does not match.")
        return

    print(f"Voter name: {record.get('name', 'Unknown')}")
    print("Starting face verification...")

    # Get full path to face image
    face_img_path = resource_path(record.get("face_image_path"))
    verified = verify_live_face(face_img_path)

    if verified:
        print("Verification complete: Access granted.")
    else:
        print("Verification failed: Access denied.")

if __name__ == "__main__":
    main()
