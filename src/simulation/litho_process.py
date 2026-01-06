import numpy as np
import yaml
from pathlib import Path
from .optics import computer_aerial_image_from_mask
from .mask import create_lines_and_spaces

"""
Lithography simulator
"""
class LithoSim:
    def __init__(self, config_path="configs/simulation_config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.shape = (self.cfg["image_size"], self.cfg["image_size"])

    def create_mask(self):
        return create_lines_and_spaces(
            shape = self.shape,
            pitch_px = self.cfg["pitch_px"],
            duty_cycle = self.cfg["duty_cycle"]
    ) 
    
    def computer_aerial_image(self, mask = None):
        if mask is None:
            mask = self.create_mask()
        aerial = computer_aerial_image_from_mask(
            mask, self.cfg["na"], self.cfg["wavelength_nm"], self.cfg["pixel_nm"]
        )
        return aerial


    def develop_resist(self, aerial_image = None):
        if aerial_image is None:
            aerial_image = self.compute_aerial_image()
            
        threshold = self.cfg["threshold"]
        resist = (aerial_image * self.cfg["dose_mj"] / 20.0 > threshold).astype(no.uint8)
        return resist

    def run_full_process(self):
        mask = self.create.mask()
        aerial = self.computer_aerial_image(mask)
        resist = self.develop_resist(aerial)
        return {"mask": mask, "aerial_image": aerial, "resist": resist}



 
