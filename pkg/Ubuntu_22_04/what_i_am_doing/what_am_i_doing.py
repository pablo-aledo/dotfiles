# Put this script in your crontab like so:
#   0 * * * * export DISPLAY=:1 ; python3 /home/your_username/Documents/what_am_i_doing/what_am_i_doing.py

from collections import OrderedDict
from tkinter import ttk
import datetime
import os.path
import psutil
import tkinter as tk

procs = [True for x in psutil.process_iter() if __file__ in " ".join(x.cmdline())]
if len(procs) > 1:
    # Another instance already running.
    exit()

# Put the txt file with the output next to this script.
filepath = os.path.dirname(__file__) + "/what_am_i_doing.txt"

window = tk.Tk()
window.attributes('-topmost', True)
window.attributes('-fullscreen', True)
window.configure(background='black')
def on_closing():
    pass  # Prevent normal closing
window.protocol("WM_DELETE_WINDOW", on_closing)
window.title("WhatAmIDoing")

frame = tk.Frame(window)
label = tk.Label(frame, text="Action:")
label.pack(side="left", fill="x")

def submit():
    if not combobox.get().strip():
        return

    with open(filepath, "a") as f:
        t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %a")
        f.write(t + ": " + combobox.get() + "\n")
    window.destroy()

def enter(event):
    submit()

window.bind('<Return>', enter)

n = tk.StringVar()
combobox = ttk.Combobox(frame, width=40, textvariable=n)

# Add previous lines to combo box.
if os.path.exists(filepath):
    with open(filepath, "r") as f:
        lines = list(f.readlines())
        lines = lines[-50:]
        lines.reverse()
        lines = [line[24:].strip() for line in lines]  # Trim away timestamp.
        lines = [line for line in lines if line]  # Skip empty.
        lines = OrderedDict.fromkeys(lines)
        lines = list(lines.keys())
        combobox["values"] = lines
combobox.pack(side="left")

button = tk.Button(frame, text="Save", command=submit)
button.pack(side="left", fill="x")

frame.place(relx=0.5, rely=0.5, anchor="center")

combobox.focus_set()
window.mainloop()