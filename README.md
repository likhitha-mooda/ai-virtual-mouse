# Real-Time AI Virtual Mouse (Hand Gesture PC Controller)

A low-latency (<25ms) touchless Human-Computer Interface (HCI) built with Python, OpenCV, and the MediaPipe Tasks API (`HandLandmarker`). Control your Windows desktop cursor, click, drag-and-drop, and scroll using a standard RGB webcam.

## Features
- **21-Point 3D Hand Tracking:** Uses MediaPipe's `HandLandmarker` in `VIDEO` mode with automatic one-time model provisioning.
- **Zero-Jitter Cursor Stabilization:** Implements a velocity-adaptive **One Euro Filter** to eliminate micro-jitter while hovering without adding lag during fast movements.
- **Knuckle-Blended Anchor Tracking:** Decouples cursor coordinates from fingertip flexion during pinch gestures so clicks never drift off-target.
- **Hysteresis Click State Machine:** Prevents accidental double-clicks using separate pinch-trigger and release thresholds.

## Gesture Controls
| Gesture | Hand Pose | Action |
| :--- | :--- | :--- |
| **Move Cursor** | Index finger raised | Smoothly moves cursor across screen |
| **Left Click** | Index up + quick Thumb-Index pinch | Triggers left click |
| **Drag & Drop** | Index up + hold Thumb-Index pinch (> 0.4s) | Holds mouse button until released |
| **Right Click** | Index + Middle fingers up, touch tips | Triggers right click |
| **Scroll** | Index + Middle + Ring fingers up | Move hand up/down to scroll |
| **Pause / Idle** | Closed fist or hand out of frame | Safely locks cursor & releases drags |

## Installation & Setup (Windows)
```bash
git clone [https://github.com/YOUR_USERNAME/ai-virtual-mouse.git](https://github.com/YOUR_USERNAME/ai-virtual-mouse.git)
cd ai-virtual-mouse
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python virtual_mouse.py