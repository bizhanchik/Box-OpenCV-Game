import tkinter as tk
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--score", type=int, required=True,default=0)
args = parser.parse_args()

def return_to_main_menu():
    root.destroy()
    subprocess.Popen([sys.executable, "main.py"])

root = tk.Tk()
root.title("Game Over")

window_width = 400
window_height = 300

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

label = tk.Label(root, text=f"Игра окончена\n\nВаш результат: {args.score}", font=("Arial", 18, "bold"))
label.pack(pady=20)

btn_main_menu = tk.Button(root, text="Вернуться в главное", command=return_to_main_menu,
                          font=("Arial", 16), width=20, height=3, bg="white", fg="black")
btn_main_menu.pack(pady=15)

root.mainloop()
