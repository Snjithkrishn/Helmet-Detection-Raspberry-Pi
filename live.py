import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time

# --- GPIO Setup ---
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# --- Load TFLite Model ---
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure input dtype and shape are compatible
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']

# --- Load Labels ---
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# --- Initialize Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

last_class = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess frame
        resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(resized, axis=0).astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        top_index = int(np.argmax(output_data))
        confidence = float(output_data[top_index])
        label = labels[top_index]

        print(f"Detected: {label} ({confidence:.2f})")
        if top_index != last_class:
            if top_index == 0:        # class 0 = helmet
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("LED ON")
                last_class = top_index
            elif top_index == 1:   # class 1 = no helmet
                GPIO.output(LED_PIN, GPIO.LOW)
                print("LED OFF")
                last_class = top_index

        # Optional display
        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
