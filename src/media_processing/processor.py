from retinaface import RetinaFace
import cv2

from enum import Enum

class ObfuscationType(str, Enum):
    BLUR = 'blur'
    PIXELATE = 'pixelate'

class Processor:
    def __init__(self, obfuscation_level: int, obfuscation_type: ObfuscationType) -> None:
        if obfuscation_level % 2 == 0:
            obfuscation_level += 1
            
        self.obfuscation_level = (obfuscation_level, obfuscation_level)
        self.obfuscation_type = obfuscation_type

    def obfuscate_faces(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        faces = self.detect_faces(image)
        image = self.obfuscate_regions(image, faces)

        return image

    def detect_faces(self, image: cv2.typing.MatLike) -> dict:
        faces = RetinaFace.detect_faces(image)
        return faces
    
    def obfuscate_regions(self, image: cv2.typing.MatLike, obfuscation_regions: dict) -> cv2.typing.MatLike:
        for region in obfuscation_regions.values():
            image = self.obfuscate_region(image, region)

        return image

    def obfuscate_region(self, image: cv2.typing.MatLike, obfuscation_region: dict) -> cv2.typing.MatLike:
        (x, y, w, h) = obfuscation_region['facial_area']
        extracted_region = image[y:h, x:w]

        if self.obfuscation_type == ObfuscationType.BLUR:
            blurred_region = cv2.GaussianBlur(extracted_region, self.obfuscation_level, 0)
            image[y:h, x:w] = blurred_region
        elif self.obfuscation_type == ObfuscationType.PIXELATE:
            height, width = extracted_region.shape[:2]

            true_obfuscation_level = 100 - self.obfuscation_level[0]
            if true_obfuscation_level <= 0:
                true_obfuscation_level = 1

            temp = cv2.resize(extracted_region, (true_obfuscation_level, true_obfuscation_level), interpolation = cv2.INTER_LINEAR)
            pixilated_region = cv2.resize(temp, (width, height), interpolation = cv2.INTER_NEAREST)
            image[y:h, x:w] = pixilated_region

        return image
    
    def obfuscate_everything(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if self.obfuscation_type == ObfuscationType.BLUR:
            image = cv2.GaussianBlur(image, self.obfuscation_level, 0)
        elif self.obfuscation_type == ObfuscationType.PIXELATE:
            height, width = image.shape[:2]

            true_obfuscation_level = 100 - self.obfuscation_level[0]
            if true_obfuscation_level <= 0:
                true_obfuscation_level = 1

            temp = cv2.resize(image, (true_obfuscation_level, true_obfuscation_level), interpolation = cv2.INTER_LINEAR)
            image = cv2.resize(temp, (width, height), interpolation = cv2.INTER_NEAREST)

        return image