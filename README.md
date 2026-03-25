# 🤖 Object Detection & Robotic Arm Grabbing

Dự án sử dụng camera + mô hình AI (TFLite) để nhận diện vật thể (chai nước) và tự động điều khiển cánh tay robot gắp vật bằng thuật toán **Gradient Descent Inverse Kinematics**.

---

## 📋 Mục lục

- [Tổng quan hệ thống](#tổng-quan-hệ-thống)
- [Giải thích thuật toán](#giải-thích-thuật-toán)
  - [1. Nhận diện vật thể (Object Detection)](#1-nhận-diện-vật-thể-object-detection)
  - [2. Chuyển đổi tọa độ Camera → Robot](#2-chuyển-đổi-tọa-độ-camera--robot)
  - [3. Bài toán Inverse Kinematics](#3-bài-toán-inverse-kinematics)
  - [4. Gradient Descent tìm góc khớp](#4-gradient-descent-tìm-góc-khớp)
  - [5. Điều khiển servo](#5-điều-khiển-servo)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu phần cứng](#yêu-cầu-phần-cứng)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Chạy chương trình](#chạy-chương-trình)
- [Tham số cấu hình](#tham-số-cấu-hình)

---

## Tổng quan hệ thống

```
Camera → TFLite Detection → Tính tọa độ thực → Chuyển hệ tọa độ → Gradient Descent IK → PWM Servo
```

Luồng xử lý chính:
1. Camera chụp ảnh liên tục
2. Mô hình TFLite (`bottle_new.tflite`) phát hiện chai nước trong khung hình
3. Dựa vào kích thước bounding box, tính khoảng cách từ camera đến vật
4. Chuyển tọa độ từ hệ camera sang hệ tọa độ robot
5. Thuật toán Gradient Descent tính góc cho 3 khớp servo
6. PCA9685 PWM driver điều khiển 5 servo của cánh tay

---

## Giải thích thuật toán

### 1. Nhận diện vật thể (Object Detection)

Sử dụng TensorFlow Lite + thư viện `tflite-support` để chạy mô hình nhận diện trên thiết bị nhúng (Raspberry Pi / Jetson). Chỉ xử lý các detection có confidence score > **0.75**.

```
score_threshold = 0.3  (lọc ban đầu của model)
confidence_threshold = 0.75  (lọc thêm trong code)
```

### 2. Chuyển đổi tọa độ Camera → Robot

Vị trí vật thể được ước lượng từ bounding box dựa trên kích thước thực của chai (~4.7 cm):

```python
scale = 640 / bbox_width               # tỉ lệ giữa frame và vật thể thực
img_hor_size = obj_size * scale         # kích thước frame thực tế (m)
distance_cam = img_hor_size / 2 / tan(hor_view_angle / 2)  # khoảng cách camera-vật
```

Camera được gắn lệch so với gốc robot: vị trí `(-0.07, -0.04, 0.21)` m, nghiêng góc **30°**. Hàm `convert_cam_to_robot()` thực hiện phép quay và tịnh tiến để ra tọa độ trong hệ robot.

### 3. Bài toán Inverse Kinematics

Cánh tay robot có **3 khớp quay** trong mặt phẳng đứng, với chiều dài:

| Khớp | Chiều dài |
|------|-----------|
| L1   | 120 mm    |
| L2   | 120 mm    |
| L3   | 90 mm     |

**Forward Kinematics** (tính vị trí đầu cánh tay từ góc khớp):

```
φ1 = a1
φ2 = a1 + a2
φ3 = a1 + a2 + a3

x = L1·cos(φ1) + L2·cos(φ2) + L3·cos(φ3)
y = L1·sin(φ1) + L2·sin(φ2) + L3·sin(φ3)
```

**Bài toán ngược (IK)**: Cho trước tọa độ mục tiêu `(x_d, y_d)`, tìm `(a1, a2, a3)` sao cho đầu cánh tay chạm đúng vị trí đó.

### 4. Gradient Descent tìm góc khớp

Định nghĩa hàm mất mát (loss function):

```
E = 0.5 × [(x - x_d)² + (y - y_d)²]
```

Tính gradient của E theo từng góc khớp:

```
∂E/∂a1 = err_x·(-L1·sin(φ1) - L2·sin(φ2) - L3·sin(φ3))
        + err_y·( L1·cos(φ1) + L2·cos(φ2) + L3·cos(φ3))

∂E/∂a2 = err_x·(-L2·sin(φ2) - L3·sin(φ3))
        + err_y·( L2·cos(φ2) + L3·cos(φ3))

∂E/∂a3 = err_x·(-L3·sin(φ3))
        + err_y·( L3·cos(φ3))
```

Cập nhật góc theo hướng ngược gradient:

```
a1 ← a1 - lr × ∂E/∂a1
a2 ← a2 - lr × ∂E/∂a2
a3 ← a3 - lr × ∂E/∂a3
```

| Tham số        | Giá trị mặc định |
|----------------|-----------------|
| `learning_rate`| 0.0001          |
| `max_iter`     | 5000            |
| `tolerance`    | 0.5 mm          |

Thuật toán dừng khi `E < 0.5 × tol²` (sai số vị trí < 0.5 mm).

Góc khớp (radian) sau đó được chuyển sang góc servo (độ):

```python
A1 = degrees(a1) + 90   # offset về vị trí trung tính
A2 = degrees(a2) + 90
A3 = degrees(a3) + 90
```

### 5. Điều khiển servo

Module `Adafruit_PCA9685` tạo tín hiệu PWM cho 5 servo qua I2C:

| Kênh | Chức năng         | Công thức PWM                        |
|------|-------------------|--------------------------------------|
| 0    | Khớp đáy (xoay)  | `120 + (180 - A1) × 430 / 180`       |
| 1    | Khớp 1           | `100 + A2 × 450 / 180`               |
| 2    | Khớp 2           | `120 + (180 - A3) × 480 / 180`       |
| 3    | Khớp 3           | `130 + A4 × 470 / 180`               |
| 4    | Kẹp (gripper)    | `400 - step5 × 100` (0=mở, 1=cắp)   |

Khi di chuyển, cánh tay được nội suy **10 bước trung gian** để chuyển động mượt mà:

```python
for i in range(1, 10):
    set_servo_angle(A1_pre + (A1 - A1_pre) / 10 * i, ...)
    time.sleep(0.03)
```

---

## Cấu trúc dự án

```
object_detect_and_grab/
├── detect.py               # Script chính: vòng lặp camera + điều khiển servo
├── utils.py                # Tiện ích: visualize, chuyển tọa độ, IK Gradient Descent
├── test_Gradient_decent.py # Unit test thuật toán Gradient Descent
├── bottle_new.tflite       # Mô hình TFLite nhận diện chai nước (custom-trained)
├── requirements.txt        # Các thư viện Python cần thiết
└── setup.sh                # Script cài đặt tự động
```

---

## Yêu cầu phần cứng

- **Raspberry Pi** (hoặc thiết bị nhúng tương đương chạy Linux)
- **Camera** (USB hoặc CSI)
- **Adafruit PCA9685** PWM driver (kết nối qua I2C)
- **5 servo motor** gắn trên cánh tay robot
- *(Tuỳ chọn)* **Google Coral Edge TPU** để tăng tốc inference

---

## Hướng dẫn cài đặt

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd object_detect_and_grab
```

### Bước 2: Cài đặt tự động

```bash
chmod +x setup.sh
./setup.sh
```

Script sẽ tự động:
- Nâng cấp pip
- Cài đặt tất cả Python dependencies từ `requirements.txt`
- Tải mô hình EfficientDet Lite0 (dự phòng)

### Bước 3: Cài đặt thủ công (nếu cần)

```bash
pip3 install -r requirements.txt
pip3 install Adafruit-PCA9685
```

### Bước 4: Kích hoạt I2C trên Raspberry Pi

```bash
sudo raspi-config
# Chọn: Interface Options → I2C → Enable
```

---

## Chạy chương trình

### Chạy với cấu hình mặc định

```bash
python3 detect.py
```

### Chạy với tham số tuỳ chỉnh

```bash
python3 detect.py \
  --model bottle_new.tflite \
  --cameraId 1 \
  --frameWidth 640 \
  --frameHeight 480 \
  --numThreads 3
```

### Test thuật toán Gradient Descent (không cần phần cứng)

```bash
python3 test_Gradient_decent.py
```

### Điều khiển khi đang chạy

| Phím    | Hành động                                |
|---------|------------------------------------------|
| `g`     | Kích hoạt chế độ gắp (grab)             |
| `ESC`   | Thoát chương trình                       |

> **Lưu ý**: Khi nhấn `g`, cánh tay sẽ gắp vật và dừng vòng lặp video.

---

## Tham số cấu hình

| Tham số             | Mặc định            | Mô tả                                |
|---------------------|---------------------|--------------------------------------|
| `--model`           | `bottle_new.tflite` | Đường dẫn tới mô hình TFLite         |
| `--cameraId`        | `1`                 | ID camera (0=built-in, 1=USB)        |
| `--frameWidth`      | `640`               | Chiều rộng khung hình (px)           |
| `--frameHeight`     | `480`               | Chiều cao khung hình (px)            |
| `--numThreads`      | `3`                 | Số luồng CPU chạy model              |
| `--enableEdgeTPU`   | `False`             | Bật tăng tốc Edge TPU (Coral)        |

Video đầu ra sẽ được lưu vào file `robot.avi` trong thư mục hiện tại.
