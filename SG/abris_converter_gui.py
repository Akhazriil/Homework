import tkinter as tk
from tkinter import filedialog, messagebox
import runpy
import sys
import os
import io


class ABRConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Конвертер ABR → XLSX")
        self.root.geometry("600x300")
        self.root.resizable(False, False)

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.getcwd())

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Конвертер файлов ABR в XLSX",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=20)

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(input_frame, text="Входной файл (.abr):", anchor="w").pack(fill=tk.X)
        input_row = tk.Frame(input_frame)
        input_row.pack(fill=tk.X)

        self.input_entry = tk.Entry(input_row, textvariable=self.input_file, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        tk.Button(input_row, text="Обзор...", command=self.browse_input, width=10).pack(side=tk.RIGHT)

        output_frame = tk.Frame(self.root)
        output_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(output_frame, text="Директория для сохранения:", anchor="w").pack(fill=tk.X)
        output_row = tk.Frame(output_frame)
        output_row.pack(fill=tk.X)

        self.output_entry = tk.Entry(output_row, textvariable=self.output_dir, width=50)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        tk.Button(output_row, text="Обзор...", command=self.browse_output, width=10).pack(side=tk.RIGHT)

        self.convert_btn = tk.Button(
            self.root,
            text="Конвертировать",
            command=self.convert,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            height=2,
            width=20
        )
        self.convert_btn.pack(pady=20)

        self.status_label = tk.Label(self.root, text="", fg="blue")
        self.status_label.pack()

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Выберите ABR файл",
            filetypes=[("ABR файлы", "*.abr"), ("Все файлы", "*.*")]
        )
        if filename:
            self.input_file.set(filename)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Выберите директорию для сохранения")
        if directory:
            self.output_dir.set(directory)

    def validate_input(self):
        input_path = self.input_file.get().strip()

        if not input_path:
            messagebox.showerror("Ошибка", "Не выбран входной файл")
            return False

        if not os.path.exists(input_path):
            messagebox.showerror("Ошибка", f"Файл не существует:\n{input_path}")
            return False

        if not input_path.lower().endswith('.abr'):
            messagebox.showerror("Ошибка", "Файл должен иметь расширение .abr")
            return False

        if not os.path.isfile(input_path):
            messagebox.showerror("Ошибка", "Выбранный путь не является файлом")
            return False

        output_path = self.output_dir.get().strip()
        if not output_path:
            messagebox.showerror("Ошибка", "Не указана директория для сохранения")
            return False

        if not os.path.exists(output_path):
            messagebox.showerror("Ошибка", f"Директория не существует:\n{output_path}")
            return False

        return True

    def convert(self):
        if not self.validate_input():
            return

        input_path = self.input_file.get().strip()
        output_dir = self.output_dir.get().strip()

        self.convert_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Конвертация...", fg="orange")
        self.root.update()

        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            script_path = os.path.join(base_path, 'abris_xl.py')

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Скрипт конвертера не найден:\n{script_path}")

            saved_argv = sys.argv
            saved_cwd = os.getcwd()
            saved_stdout = sys.stdout
            sys.argv = ['abris_xl.py', input_path]
            sys.stdout = io.StringIO()
            os.chdir(output_dir)

            try:
                runpy.run_path(script_path, run_name='__main__')
            except SystemExit as e:
                if e.code and e.code != 0:
                    output = sys.stdout.getvalue()
                    raise RuntimeError(f"Ошибка конвертации:\n{output or 'Неизвестная ошибка'}")
            finally:
                sys.argv = saved_argv
                sys.stdout = saved_stdout
                os.chdir(saved_cwd)

            output_file = os.path.join(output_dir, 'lesoseki.xlsx')

            if os.path.exists(output_file):
                self.status_label.config(text="Успешно конвертировано!", fg="green")
                messagebox.showinfo(
                    "Успех",
                    f"Файл успешно конвертирован!\n\nСохранено: {output_file}"
                )
            else:
                raise FileNotFoundError("Выходной файл не был создан")

        except Exception as e:
            self.status_label.config(text="✗ Ошибка", fg="red")
            messagebox.showerror("Ошибка конвертации", str(e))

        finally:
            self.convert_btn.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = ABRConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
