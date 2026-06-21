from ultralytics import YOLO
import cv2

modelo = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    resultado = modelo(frame)

    quantidade = 0

    for r in resultado:
        for box in r.boxes:
            classe = int(box.cls)
            nome = modelo.names[classe]

            if nome == "person":
                quantidade += 1

    cv2.putText(
        frame,
        f"Pessoas: {quantidade}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "VisionCounter",
        frame
    )

    if cv2.waitKey(1)==27:
        break

camera.release()
cv2.destroyAllWindows()
