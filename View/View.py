from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from Controller.Controller import Controller


class View:

    def __init__(self):
        self.root = Tk()
        self.define_root()
        self.controller = Controller()
        self.analyze_speaker_frame = self.create_analyze_speaker_frame()
        self.add_speaker_frame = self.create_add_speaker_frame()
        self.speakerlist_frame = self.create_speakerlist_frame()
        self.show_add_speaker_frame()
        self.root.mainloop()

    def define_root(self):
        self.root.title('Speaker recognition')
        self.root.iconbitmap('icon/test.ico')
        self.root.geometry("600x400")
        # create menu bar
        my_menu = Menu(self.root)
        self.root.config(menu=my_menu)
        # create menu
        options_menu = Menu(my_menu)
        my_menu.add_cascade(label="options", menu=options_menu)
        options_menu.add_command(label="add speaker", command=self.show_add_speaker_frame)
        options_menu.add_command(label="analyze speaker", command=self.show_analyze_speaker_frame)
        options_menu.add_command(label="speakerlist", command=self.show_speakerlist_frame)

    # logic to change frame
    def show_analyze_speaker_frame(self):
        self.hide_all_frames()
        self.analyze_speaker_frame.pack(fill="both", expand=1)

    def show_speakerlist_frame(self):
        self.hide_all_frames()
        self.speakerlist_frame.pack(fill="both", expand=1)

    def show_add_speaker_frame(self):
        self.hide_all_frames()
        self.add_speaker_frame.pack(fill="both", expand=1)

    def hide_all_frames(self):
        self.add_speaker_frame.pack_forget()
        self.analyze_speaker_frame.pack_forget()
        self.speakerlist_frame.pack_forget()

    def open_browse_window(self):
        input_audio = filedialog.askopenfilenames(parent=self.add_speaker_frame, title='Choose an audio file')
        self.controller.input_audio = input_audio

    def validate_speaker(self):
        self.controller.validate_speaker()

    def open_browse_window_folder(self):
        folder = filedialog.askdirectory(parent=self.add_speaker_frame, title='Choose an audio file')
        self.controller.folder = folder

    def add_speaker(self):
        self.controller.add_speaker()

    def create_add_speaker_frame(self):
        add_speaker_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(add_speaker_frame, text="add speaker")
        label_top.pack(pady=20)

        btn_browse = Button(add_speaker_frame, text="browse", command=self.open_browse_window_folder)
        btn_browse.pack(pady=10)

        input_fname = Entry(add_speaker_frame)
        input_fname.insert(0, "Enter the first name")
        input_fname.pack(pady=10)
        input_lname = Entry(add_speaker_frame)
        input_lname.insert(0, "Enter the last name")
        input_lname.pack(pady=10)

        btn_train = Button(add_speaker_frame, text="training", command=self.add_speaker)
        btn_train.pack(pady=10)
        return add_speaker_frame

    def create_analyze_speaker_frame(self):
        analyze_speaker_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(analyze_speaker_frame, text="analyze speaker")
        label_top.pack(pady=20)
        btn_browse = Button(analyze_speaker_frame, text="browse", command=self.open_browse_window)
        btn_browse.pack(pady=10)
        btn_analyze = Button(analyze_speaker_frame, text="analyze", command=self.validate_speaker())
        btn_analyze.pack(pady=10)
        return analyze_speaker_frame

    def create_speakerlist_frame(self):
        speakerlist_frame = Frame(self.root, width=400, height=400, bg="white")
        label_top = Label(speakerlist_frame, text="speakerlist")
        label_top.pack(pady=20)
        # create Treeviw
        my_tree = ttk.Treeview(speakerlist_frame)
        # define columns
        my_tree['columns'] = ('ID', 'Vorname', 'Nachname')
        # format columns
        my_tree.column("#0", width=0, minwidth=25)
        my_tree.column("ID", anchor=CENTER, width=80, minwidth=25)
        my_tree.column("Vorname", anchor=W, width=120, minwidth=25)
        my_tree.column("Nachname", anchor=W, width=120, minwidth=25)
        my_tree.column("ID", anchor=CENTER, width=120, minwidth=25)
        # create headings
        my_tree.heading("#0", text="", anchor=W)
        my_tree.heading("ID", text="ID", anchor=CENTER)
        my_tree.heading("Vorname", text="Vorname", anchor=W)
        my_tree.heading("Nachname", text="Nachname", anchor=W)

        # add data
        my_tree.insert(parent='', index='end', text="", values=(1, "Test", "Daten"))
        my_tree.insert(parent='', index='end', text="", values=(2, "Jonas", "Feige"))
        my_tree.pack(pady=20)
        return speakerlist_frame


if __name__ == '__main__':
    View()
