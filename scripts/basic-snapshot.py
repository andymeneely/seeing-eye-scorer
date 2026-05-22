import cv2
import time

cam = cv2.VideoCapture(0)

LOW_LIGHT_MODE = True
TARGET_GAIN = 32


def sharpness_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def capture_best_frame(camera, num_frames=20):
    best_frame = None
    best_score = -1.0

    for _ in range(num_frames):
        ok, candidate = camera.read()
        if not ok:
            print("Warning: Failed to read frame from webcam")
            continue

        score = sharpness_score(candidate)
        if score > best_score:
            best_score = score
            best_frame = candidate

    return best_frame, best_score

# 1. Set camera resolution to 4K (3840x2160)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Get the actual resolution being used
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution set to: {width}x{height}")

# Low-light tuning: gain/exposure
if LOW_LIGHT_MODE:
    gain_supported = cam.set(cv2.CAP_PROP_GAIN, TARGET_GAIN)
    if gain_supported:
        actual_gain = cam.get(cv2.CAP_PROP_GAIN)
        print(f"Gain set to: {actual_gain:.1f}")
    else:
        print("Gain control not supported by this camera/backend")

    auto_exposure_supported = cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    if auto_exposure_supported:
        print("Auto exposure enabled for low light")
    else:
        print("Auto exposure control not supported by this camera/backend")

# 2. Control camera focus
# Enable autofocus (if supported by your camera)
autofocus_enabled = cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
if autofocus_enabled:
    print("Autofocus enabled")
else:
    print("Autofocus control not supported; falling back to current camera settings")

# Let exposure/focus settle before capturing
print("Warming up the camera...")
time.sleep(1.0)

# Capture using autofocus first
auto_frame, auto_score = capture_best_frame(cam, num_frames=25)
print(f"Best autofocus sharpness: {auto_score:.1f}")

# Fallback: manual focus sweep (many webcams use 0-255)
manual_frame = None
manual_score = -1.0
manual_focus_value = None

autofocus_off = cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
if autofocus_off:
    print("Running manual focus sweep...")
    for focus in [0, 32, 64, 96, 128, 160, 192, 224, 255]:
        cam.set(cv2.CAP_PROP_FOCUS, focus)
        time.sleep(0.15)
        candidate, score = capture_best_frame(cam, num_frames=4)
        print(f"Focus {focus:3d} -> sharpness {score:.1f}")
        if score > manual_score:
            manual_score = score
            manual_frame = candidate
            manual_focus_value = focus
else:
    print("Manual focus control not supported by this camera/backend")

# Choose the sharpest result available
if manual_score > auto_score and manual_frame is not None:
    best_frame = manual_frame
    best_score = manual_score
    ret = True
    print(f"Using manual focus value {manual_focus_value}")
else:
    best_frame = auto_frame
    best_score = auto_score
    ret = auto_frame is not None



# 3. If the frame was captured successfully, save it
if ret:
    cv2.imwrite("webcam_capture.jpg", best_frame)
    print(f"Image captured successfully! Sharpness score: {best_score:.1f}")
else:
    print("Error: Could not access the webcam.")

# 4. Release the camera resource
cam.release()
