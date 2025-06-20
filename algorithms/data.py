import struct
import os

values = (40, 30, 30, 1, 3, 3, 0)

# Ensure the directory exists
output_dir = "../trandata"
os.makedirs(output_dir, exist_ok=True)

# Correct full path
file_path = os.path.join(output_dir, "slice_ctrl.bin")

# Write the binary data
with open(file_path, 'wb') as f:
    f.write(struct.pack('7i', *values))

print(f"Created {file_path} with values {values}")
