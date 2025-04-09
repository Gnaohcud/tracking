import os

# Thư mục chứa ảnh
folder_path = "output_frames_9_4_4"

# Lấy danh sách file ảnh (lọc file PNG hoặc JPG nếu cần)
image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

# Đổi tên từng file
for i, filename in enumerate(image_files):
    ext = os.path.splitext(filename)[1]  # phần đuôi file (.png, .jpg, ...)
    new_name = f"frame_9_4_4_{i}{ext}"
    
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)

print(f"✅ Đã đổi tên {len(image_files)} ảnh trong thư mục '{folder_path}'.")
