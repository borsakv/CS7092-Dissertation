import cv2
from media_processing.processor import Processor, ObfuscationType

class ImageMedia(Processor):
    def __init__(self, image_file_path: str, save_path: str, obfuscation_level: int, obfuscation_type: ObfuscationType) -> None:
        super().__init__(obfuscation_level, obfuscation_type)

        self.image_file_path = image_file_path
        self.save_path = save_path

        self.image = cv2.imread(self.image_file_path)

    def save_image(self) -> None:
        cv2.imwrite(self.save_path, self.image)

    def obfuscate_image_faces(self) -> None:
        self.image = self.obfuscate_faces(self.image)

        self.save_image()

    def obfuscate_full_image(self) -> None:
        self.image = self.obfuscate_everything(self.image)

        self.save_image()