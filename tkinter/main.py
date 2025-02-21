import tkinter as tk
import subprocess
import sys

selected_camera = 0

def start_game(cam):
    global selected_camera
    selected_camera = cam
    print(f"Выбрана камера: {selected_camera}")
    root.destroy()
    subprocess.Popen([sys.executable, "game_selection.py", str(selected_camera)])

root = tk.Tk()
root.title("Выбор камеры")

window_width = 600
window_height = 400

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

label = tk.Label(root, text="Выберите камеру:", font=("Arial", 18, "bold"))
label.pack(pady=20)

btn_iphone = tk.Button(root, text="📱 Камера iPhone", command=lambda: start_game(0),
                       font=("Arial", 16), width=20, height=3, bg="white", fg="black")
btn_iphone.pack(pady=15)

btn_macbook = tk.Button(root, text="💻 Камера MacBook", command=lambda: start_game(1),
                        font=("Arial", 16), width=20, height=3, bg="white", fg="black")
btn_macbook.pack(pady=15)

root.mainloop()
