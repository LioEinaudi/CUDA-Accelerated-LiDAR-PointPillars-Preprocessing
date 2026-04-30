import os
import random
import struct


os.makedirs("data", exist_ok=True)

N = 120000
filepath = "data/000000.bin"

print(f"正在生成 {N} 个点，请稍候...")


with open(filepath, "wb") as f:
    for _ in range(N):
    
        x = random.uniform(-50.0, 100.0)
        y = random.uniform(-50.0, 50.0)
        z = random.uniform(-5.0, 5.0)
        intensity = random.uniform(0.0, 1.0)
        
        
        f.write(struct.pack('<4f', x, y, z, intensity))

print(f"成功生成零依赖假数据: {filepath}！")