from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from tensorflow.python.framework.errors_impl import InvalidArgumentError

from controller.Controller import Controller


class View:

    def __init__(self):
        self.root = Tk()
        self.define_root()
        self.my_tree = None
        self.output_text = None
        self.output_text_path = None
        self.output_text_add_speaker = None
        self.add_speaker_frame = None
        self.controller = Controller()
        self.analyze_speaker_frame = None
        self.create_analyze_speaker_frame()
        self.input_lname = None
        self.input_fname = None
        self.create_add_speaker_frame()
        self.speakerlist_frame = self.create_speakerlist_frame()
        self.show_add_speaker_frame()
        self.root.mainloop()

    def define_root(self):
        self.root.title('Speaker recognition')
        self.root.iconbitmap('../icon/test.ico')
        self.root.geometry("600x400")
        # create menu bar
        my_menu = Menu(self.root)
        self.root.config(menu=my_menu)
        # create menu
        options_menu = Menu(my_menu)
        my_menu.add_cascade(label="menu", menu=options_menu)
        options_menu.add_command(label="add speaker", command=self.show_add_speaker_frame)
        options_menu.add_command(label="analyze speaker", command=self.show_analyze_speaker_frame)
        options_menu.add_command(label="speakerlist", command=self.show_speakerlist_frame)

    # logic to change frame
    def show_analyze_speaker_frame(self):
        self.hide_all_frames()
        self.analyze_speaker_frame.pack(fill="both", expand=1)

    def show_speakerlist_frame(self):
        self.get_speaker_list_data()
        self.hide_all_frames()
        self.speakerlist_frame.pack(fill="both", expand=1)

    def show_add_speaker_frame(self):
        self.hide_all_frames()
        self.add_speaker_frame.pack(fill="both", expand=1)

    def hide_all_frames(self):
        self.add_speaker_frame.pack_forget()
        self.analyze_speaker_frame.pack_forget()
        self.speakerlist_frame.pack_forget()

    def open_browse_window_add_audio_path(self):
        self.output_text_path.delete(1.0, END)
        self.output_text.delete(1.0, END)
        input_audio_path = filedialog.askopenfilenames(parent=self.add_speaker_frame, title='Choose an audio file')
        if input_audio_path != "":
            self.controller.input_audio_path = input_audio_path[0]
            text = "Dateipfad: " + input_audio_path[0]

        else:
            text = "Bitte eine Datei auswählen"
        self.output_text_path.insert(1.0, text)
        self.output_text_path.pack()

    def validate_speaker(self):
        self.output_text.delete(1.0, END)
        try:
            predicted_speaker = self.controller.validate_speaker()
            text = "Progonstizierter Sprecher: " + predicted_speaker
        except InvalidArgumentError:
            text = "Bitte eine waveform-Datei hinzufügen"
        self.output_text.insert(1.0, text)
        self.output_text.pack()

    def open_browse_window_folder(self):
        folder = filedialog.askdirectory(parent=self.add_speaker_frame, title='Choose an audio file')
        self.controller.folder = folder

    def train_model(self):
        self.output_text_add_speaker.delete(1.0, END)
        self.output_text_add_speaker.insert(1.0, "Das Modell trainiert. Das kann einige Minuten dauern...")
        self.controller.train_model()
        self.output_text_add_speaker.delete(1.0, END)
        self.output_text_add_speaker.insert(1.0, "Das trainierte Modell steht zur Verfügung.")

    def add_speaker(self):
        self.output_text_add_speaker.delete(1.0, END)
        try:
            self.controller.FIRST_NAME = self.input_fname.get()
            self.controller.LAST_NAME = self.input_lname.get()
            self.controller.add_speaker()
            text = "Der Sprecher: " + self.input_fname.get() + " " + self.input_lname.get() + " wurde hinzugefügt."
            self.output_text_add_speaker.insert(1.0, text)
            self.output_text_add_speaker.pack()
        except FileExistsError as arr:
            self.output_text_add_speaker.insert(1.0, arr)
            self.output_text_add_speaker.pack()
        except TypeError:
            self.output_text_add_speaker.insert(1.0, "Bitte einen Ordner auswählen.")
            self.output_text_add_speaker.pack()

    def create_add_speaker_frame(self):
        self.add_speaker_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(self.add_speaker_frame, text="add speaker")
        label_top.pack(pady=20)

        btn_browse = Button(self.add_speaker_frame, text="browse", command=self.open_browse_window_folder)
        btn_browse.pack(pady=10)

        self.input_fname = Entry(self.add_speaker_frame)
        self.input_fname.insert(0, "Enter the first name")
        self.input_fname.pack(pady=10)

        self.input_lname = Entry(self.add_speaker_frame)
        self.input_lname.insert(0, "Enter the last name")
        self.input_lname.pack(pady=10)

        btn_add_speaker = Button(self.add_speaker_frame, text="add speaker", command=self.add_speaker)
        btn_add_speaker.pack(pady=10)

        btn_train = Button(self.add_speaker_frame, text="training", command=self.train_model)
        btn_train.pack(pady=10)
        self.output_text_add_speaker = Text(self.add_speaker_frame, height=10)

    def create_analyze_speaker_frame(self):
        self.analyze_speaker_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(self.analyze_speaker_frame, text="analyze speaker")
        label_top.pack(pady=20)
        btn_browse = Button(self.analyze_speaker_frame, text="browse", command=self.open_browse_window_add_audio_path)
        btn_browse.pack(pady=10)
        self.output_text_path = Text(self.analyze_speaker_frame, height=10)
        btn_analyze = Button(self.analyze_speaker_frame, text="analyze", command=self.validate_speaker)
        btn_analyze.pack(pady=10)
        self.output_text = Text(self.analyze_speaker_frame, height=10)

    def create_speakerlist_frame(self):
        speakerlist_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(speakerlist_frame, text="speakerlist")
        label_top.pack(pady=20)
        # create Treeview
        self.my_tree = ttk.Treeview(speakerlist_frame)
        # define columns
        self.my_tree['columns'] = ('ID', 'Vorname', 'Nachname')
        # format columns
        self.my_tree.column("#0", width=0, minwidth=25)
        self.my_tree.column("ID", anchor=CENTER, width=80, minwidth=25)
        self.my_tree.column("Vorname", anchor=W, width=120, minwidth=25)
        self.my_tree.column("Nachname", anchor=W, width=120, minwidth=25)
        self.my_tree.column("ID", anchor=CENTER, width=120, minwidth=25)
        # create headings
        self.my_tree.heading("#0", text="", anchor=W)
        self.my_tree.heading("ID", text="ID", anchor=CENTER)
        self.my_tree.heading("Vorname", text="Vorname", anchor=W)
        self.my_tree.heading("Nachname", text="Nachname", anchor=W)

        return speakerlist_frame

    def get_speaker_list_data(self):
        for i in self.my_tree.get_children():
            self.my_tree.delete(i)

        names_list = self.controller.class_names
        for index in range(len(names_list)):
            splitted = names_list[index].split("_")
            self.my_tree.insert(parent='', index='end', text="", values=(index, splitted[0], splitted[1]))
            self.my_tree.pack(pady=20)


if __name__ == '__main__':
    View()
