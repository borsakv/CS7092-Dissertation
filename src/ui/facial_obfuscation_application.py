import customtkinter as ctk
from tkinter import filedialog
from CTkListbox import CTkListbox
from CTkMessagebox import CTkMessagebox
from PIL import Image

import math
from pymediainfo import MediaInfo
from pathlib import Path

from media_processing.image_media import ImageMedia
from media_processing.video_media import VideoMedia
from media_processing.processor import ObfuscationType

WINDOW_SIZE = '1280x720'
DEFAULT_OBFUSCATION = 20

class FacialObfuscationApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.iconbitmap('src/ui/data/application_logo.ico')
        self.title('Face Obfuscation Application')
        self.geometry(WINDOW_SIZE)
        self.minsize(1280, 720)
        self.maxsize(1280, 720)
        
        # Set theme and color
        ctk.set_appearance_mode('dark')  
        ctk.set_default_color_theme('blue')
        
        # Create a grid system
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure((0, 1, 2), weight = 1)

        # Set initial values
        self.obfuscation_level = DEFAULT_OBFUSCATION
        self.obfuscation_type = ObfuscationType.BLUR
        self.obfuscate_everything = False
        self.file_paths = []
        self.save_target_dir = ''

        # Create button to select files
        self.upload_button = ctk.CTkButton(self, text = 'Upload Files', command = self.upload_files)
        self.upload_button.grid(row = 0, column = 0, columnspan = 3, padx = 5, pady = 10, sticky = 'nsew')

        # Create 'selected files' label
        self.selected_files_label = ctk.CTkLabel(self, text = 'Selected Files:')
        self.selected_files_label.grid(row = 1, column = 0, padx = 5, pady = (20, 0), sticky = 'w')

        # Create listbox to display selected files
        self.file_list = CTkListbox(self, multiple_selection = True)
        self.file_list.grid(row = 2, column = 0, rowspan = 4, padx = 5, sticky = 'nsew')

        # Create 'settings' label
        self.settings_label = ctk.CTkLabel(self, text = 'Settings:')
        self.settings_label.grid(row = 1, column = 1, pady = (20, 0), sticky = 'w')

        # Create 'obfuscation level' label
        self.obfuscation_level_label = ctk.CTkLabel(self, text = 'Obfuscation Level')
        self.obfuscation_level_label.grid(row = 2, column = 1, sticky = 'nsew')

        # Create label to display the obfuscation level image
        self.image_label = ctk.CTkLabel(self, text = '')
        self.display_obfuscation_level_img()

        # Create slider for obfuscation level
        self.obfuscation_level_slider = ctk.CTkSlider(self, from_ = 0, to = 100, command = self.obfuscation_level_slider_moved)
        self.obfuscation_level_slider.set(self.obfuscation_level)
        self.obfuscation_level_slider.grid(row = 3, column = 2, sticky = 'nsew', pady = (10, 0))

        # Create 'obfuscation type' label
        self.obfuscation_type_label = ctk.CTkLabel(self, text = 'Obfuscation Type')
        self.obfuscation_type_label.grid(row = 4, column = 1, pady = (20, 0), sticky = 'nsew')

        # Create combobox for obfuscation type
        self.obfuscation_type_combobox = ctk.CTkComboBox(self, values = ['Gaussian Blur', 'Pixelation'], state = 'readonly', command = self.obfuscation_type_combobox_value_changed)
        self.obfuscation_type_combobox.set('Gaussian Blur')
        self.obfuscation_type_combobox.grid(row = 4, column = 2, padx = 5, pady = (20, 0), sticky = 'nsew')

        # Create 'obfuscate full image' label
        self.obfuscate_full_image_label = ctk.CTkLabel(self, text = 'Obfuscate Full Image')
        self.obfuscate_full_image_label.grid(row = 5, column = 1, pady = (20, 0), sticky = 'nsew')

        # Create switch for full image obfuscation
        self.obfuscate_everything_switch = ctk.CTkSwitch(self, text = '', command = self.obfuscate_everything_switch_toggle)
        self.obfuscate_everything_switch.grid(row = 5, column = 2, pady = (20, 0), sticky = 'nsew')

        # Create textbox to display save target directory
        self.save_target_dir_text_box = ctk.CTkTextbox(self, height = 30, fg_color = 'grey', border_color = 'black', text_color = 'black', border_width = 2)
        self.save_target_dir_text_box.tag_config('right', justify = 'right')
        self.save_target_dir_text_box.insert('0.0', 'Results Directory', 'right')
        self.save_target_dir_text_box.configure(state = 'disabled')
        self.save_target_dir_text_box.grid(row = 6, column = 0, padx = 5, pady = 10, sticky = 'new')

        # Create button to browse save target directory
        self.browse_save_target_dir_button = ctk.CTkButton(self, height = 30, text = 'Browse', text_color = 'black', fg_color = 'grey', command = self.browse_save_target_dir)
        self.browse_save_target_dir_button.grid(row = 6, column = 1, padx = 5, pady = 10, sticky = 'new')

        # Create button to process selected images and save the results to the selected directory
        self.run_obfuscation_button = ctk.CTkButton(self, height = 30, text = 'Run Obfuscation', command = self.run_obfuscation)
        self.run_obfuscation_button.grid(row = 6, column = 2, padx = 5, pady = 10, sticky = 'new')

    # Handlers

    def upload_files(self):
        self.file_paths = []
        self.file_list.delete(0, 'end')

        self.file_paths = filedialog.askopenfilenames()

        if self.file_paths:
            for file_path in self.file_paths:
                self.file_list.insert('end', file_path)

    def display_obfuscation_level_img(self):
        image = Image.open(f'src/ui/data/{self.obfuscation_type.value}_level_{self.obfuscation_level}.png')
        photo = ctk.CTkImage(light_image = image, dark_image = image, size = (254, 271))

        # If the label has not been packed (displayed), do it now
        if not self.image_label.winfo_ismapped():
            self.image_label.grid(row = 2, column = 2, sticky = 'nsew')

        # Set the image in the label
        self.image_label.configure(image = photo)
        self.image_label.image = photo  # Keep a reference so it's not garbage collected

    def obfuscation_level_slider_moved(self, value):
        self.obfuscation_level = math.ceil(value / 5.0) * 5
        self.display_obfuscation_level_img()

    def obfuscation_type_combobox_value_changed(self, value):
        if value == 'Gaussian Blur':
            self.obfuscation_type = ObfuscationType.BLUR
        if value == 'Pixelation':
            self.obfuscation_type = ObfuscationType.PIXELATE

        self.display_obfuscation_level_img()

    def obfuscate_everything_switch_toggle(self):
        state =  self.obfuscate_everything_switch.get()

        if state == 1:
            self.obfuscate_everything = True
        elif state == 0:
            self.obfuscate_everything = False

    def browse_save_target_dir(self):
        self.save_target_dir = ''

        self.save_target_dir = filedialog.askdirectory()

        if self.save_target_dir:
            self.save_target_dir_text_box.configure(state = 'normal')
            self.save_target_dir_text_box.delete('0.0', 'end')
            self.save_target_dir_text_box.insert('0.0', self.save_target_dir, 'right')
            self.save_target_dir_text_box.configure(state = 'disabled')

    def run_obfuscation(self):
        if len(self.file_paths) == 0:
            CTkMessagebox(title = 'Error', message = 'Please Upload File(s) to Process', icon = 'cancel')
        elif self.save_target_dir == '':
            CTkMessagebox(title = 'Error', message = 'Please Select a Directory to Save Results', icon = 'cancel')

        for file_path in self.file_paths:
            fileInfo = MediaInfo.parse(file_path)
            file_name = Path(file_path).stem

            for track in fileInfo.tracks:
                if track.track_type == 'Video':
                    save_file_name = f'{self.save_target_dir}/obfuscated_{file_name}.avi'
                    video_media_processor = VideoMedia(file_path, save_file_name, self.obfuscation_level, self.obfuscation_type)
                    if self.obfuscate_everything:
                        video_media_processor.obfuscate_full_video()
                    else:
                        video_media_processor.obfuscate_video_faces()
                elif track.track_type == 'Image':
                    save_file_name = f'{self.save_target_dir}/obfuscated_{file_name}.png'
                    image_media_processor = ImageMedia(file_path, save_file_name, self.obfuscation_level, self.obfuscation_type)
                    if self.obfuscate_everything:
                        image_media_processor.obfuscate_full_image()
                    else:
                        image_media_processor.obfuscate_image_faces()
        