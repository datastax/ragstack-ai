import csv
import os
from typing import List

Embedding = List[List[float]]


class TestData:
    @staticmethod
    def _get_test_data_path(file_name: str) -> str:
        script_directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_directory, "test_data", file_name)

    @staticmethod
    def _get_text_file(file_name: str) -> str:
        with open(TestData._get_test_data_path(file_name)) as f:
            return f.read()

    @staticmethod
    def _get_csv_embedding(csv_file_name: str) -> Embedding:
        with open(TestData._get_test_data_path(csv_file_name)) as f:
            reader = csv.reader(f)
            return [[float(value) for value in row] for row in reader]

    @staticmethod
    def save_csv_embedding(csv_file_name: str, embedding: Embedding) -> None:
        with open(TestData._get_test_data_path(csv_file_name), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(embedding)

    @staticmethod
    def climate_change_text() -> str:
        """Returns: A short, highly-technical text on climate change."""
        return TestData._get_text_file("climate_change.txt")

    @staticmethod
    def climate_change_embedding() -> Embedding:
        """Returns: An embedding for the `climate_change_text()` text"""
        return TestData._get_csv_embedding("climate_change.csv")

    @staticmethod
    def marine_animals_text() -> str:
        """Returns:
            A story of approx 350 words about marine animals.

        Potential queries on the text:
            - What kind of fish lives in shallow coral reefs?
            - What communication methods do dolphins use within their pods?
            - How do anglerfish adapt to the deep ocean's darkness?
            - What role do coral reefs play in marine ecosystems?
        """
        return TestData._get_text_file("marine_animals.txt")

    @staticmethod
    def nebula_voyager_text() -> str:
        """Returns:
            A story of approx 2500 words about a theoretical spaceship.
            Includes very technical names and terms that can be
            difficult for standard retrieval systems.

        Potential queries on the text:
            - Who developed the Astroflux Navigator?
            - Describe the phenomena known as "Chrono-spatial Echoes"?
            - What challenges does the Quantum Opacity phenomenon present to the crew of
              the Nebula Voyager?
            - How does the Bioquantum Array aid Dr. Nyx Moreau in studying the
              Nebuloforms within Orion's Whisper?
            - What are Xenospheric Particulates?
            - What is the significance of the Cryptolingual Synthesizer used by Jiro
              Takashi, and how does it function?
        """
        return TestData._get_text_file("nebula_voyager.txt")

    @staticmethod
    def renewable_energy_text() -> str:
        """Returns: A short, highly-technical text on renewable energy"""
        return TestData._get_text_file("renewable_energy.txt")

    @staticmethod
    def renewable_energy_embedding() -> Embedding:
        """Returns: An embedding for the `renewable_energy_text()` text"""
        return TestData._get_csv_embedding("renewable_energy.csv")
