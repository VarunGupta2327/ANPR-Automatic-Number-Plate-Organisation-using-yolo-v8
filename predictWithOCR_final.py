import cv2
import os
import torch
import easyocr
from ultralytics import YOLO


# ----------------------------
# OCR FUNCTION
# ----------------------------
reader = easyocr.Reader(['en'])

def get_ocr_text(image, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    if len(results) == 0:
        return ""

    # pick highest-confidence text
    best = max(results, key=lambda x: x[2]),
    return best[1]


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    # ----------- SETTINGS -----------
    model_path = "yolov8n.pt"      # Your trained model or yolov8n.pt
    source_path = r"C:\Users\varun.LAPTOP-JI5C1818\Desktop\ANPR\Car1.png"    # Can be image OR video
    # --------------------------------

    # Resolve absolute path
    source_path = os.path.abspath(source_path)
    print("\nLoading source:", source_path)

    if not os.path.exists(source_path):
        print("❌ ERROR: Source file does not exist.")
        return

    # Load YOLO model
    model = YOLO(model_path)

    # Check if source is image or video
    ext = os.path.splitext(source_path)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png"]

    if is_image:
        # --------------------- IMAGE MODE -------------------------
        frame = cv2.imread(source_path)

        if frame is None:
            print("❌ Cannot open image.")
            return

        results = model(frame)[0]

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)

            text = get_ocr_text(frame, xyxy)
            label = text if text != "" else f"{model.names[cls]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO + OCR", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    else:
        # --------------------- VIDEO MODE -------------------------
        cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            print("❌ ERROR: Cannot open video source.")
            return

        print("✅ Video opened successfully.\nPress Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)

                text = get_ocr_text(frame, xyxy)
                label = text if text != "" else f"{model.names[cls]} {conf:.2f}"

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("YOLO + OCR", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
