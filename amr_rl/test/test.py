import matplotlib.pyplot as plt
import numpy as np

def draw_divided_circle(parts):
    # 建立一個新的畫布
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 設定圓的參數
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    # 畫出外圓
    ax.plot(x_circle, y_circle, color='black', linewidth=2)
    
    # 計算並畫出分割線
    # 每一份的角度為 2π / parts
    angles = np.linspace(0, 2 * np.pi, parts, endpoint=False)
    
    for angle in angles:
        # 從圓心 (0,0) 連線到圓周上的點 (cosθ, sinθ)
        x_line = [0, np.cos(angle)]
        y_line = [0, np.sin(angle)]
        ax.plot(x_line, y_line, color='blue', linestyle='--')

    # 設定圖表格式，確保圓形比例正確
    ax.set_aspect('equal')
    plt.title(f"Circle divided into {parts} equal parts")
    plt.axis('off')  # 隱藏座標軸
    plt.show()

# 使用者輸入
try:
    x = int(input("請輸入要切分的等分數 (例如 6): "))
    if x > 0:
        draw_divided_circle(x)
    else:
        print("請輸入大於 0 的整數。")
except ValueError:
    print("輸入無效，請輸入數字。")