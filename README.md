# Helmet Detection on Raspberry Pi using TFLite + GPIO

This project detects whether a person is wearing a helmet using a Raspberry Pi, a quantized TFLite model, and toggles an LED via GPIO based on the prediction.

---

## Live Detection Features

- Real-time webcam-based detection using a quantized `.tflite` model
- Turns **ON an LED** (connected to GPIO 17) when a **helmet is detected**
- Designed for edge inference on Raspberry Pi (low power, fast)
- Uses OpenCV for video stream, TFLite Runtime for inference, and RPi.GPIO for LED control

---

## Model Details

- Format: TensorFlow Lite (quantized)
- Input shape: `(1, 224, 224, 3)`
- Output: 2-class logits
- Labels (`labels.txt`):

```
0 Helmet
1 No Helmet
```

The model predicts either class 0 or class 1 with confidence scores in the range `[0, 255]`.

---

## Requirements

### Hardware:
- Raspberry Pi 3B/4 (tested)
- USB Webcam or Pi Camera
- LED + Resistor (~220Ω)
- Breadboard & Jumper Wires

### Python Libraries:

Recommended versions that are known to work reliably:

```
numpy==1.24.4
tflite-runtime==2.13.0
opencv-python==4.8.0.76
RPi.GPIO==0.7.1
```

> ⚠️ `tflite-runtime` is NOT compatible with NumPy ≥ 2.0. You MUST use NumPy 1.24.x.

To install everything:

```bash
pip uninstall numpy tflite-runtime -y
pip install numpy==1.24.4
pip install tflite-runtime
pip install opencv-python==4.8.0.76
pip install RPi.GPIO==0.7.1
```
---

## GPIO Pin Configuration

- **GPIO 17** is used to drive the LED.
- Connect the long leg (anode) of the LED to GPIO 17 through a 220Ω resistor.
- Connect the short leg (cathode) to GND.

---

## How to Run

Make sure your camera is connected and enabled.

```bash
python3 live.py
```

To exit:
- Press `Ctrl + C` in the terminal, or
- Press `q` in the OpenCV window.

---

## Inference Output (Terminal Example)

```text
Detected: No Helmet (255.00)
LED OFF
Detected: Helmet (255.00)
LED ON
```

---

## Tips & Customization

- To change the GPIO pin, update `LED_PIN = 17` in `live.py`.
- You can use `cv2.resize()` or change `cap.set()` to tweak input resolution.
- Add buzzer or relay control using additional GPIOs.

---

## License

MIT License

---

## Author

**Sanjith R**  
Feel free to open issues or contribute!
