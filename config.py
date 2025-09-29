import pprint
import csv
import os


class Config:
    """
        Configuration class for Radon Transform analysis.

        Reads parameters from a CSV file and provides default values if needed.
        """

    def __init__(self, input_config_csv=None, **kwargs):
        """
        Initialize configuration from a CSV file or keyword arguments.

        Args:
            input_config_csv (str): Path to CSV file containing config parameters.
            **kwargs: Default values if CSV is not provided or lacks keys.
        """
        self.config = {}

        # Load from CSV if provided
        if input_config_csv:
            self._load_from_csv(input_config_csv)

        # Override with kwargs (higher priority than CSV)
        self.config.update(kwargs)

        # Set defaults if keys are still missing
        self._set_defaults()

    def _load_from_csv(self, csv_filename):
        """Read configuration parameters from a key-value CSV file."""
        try:
            csv_dir = os.path.join(os.path.dirname(__file__), "csv")
            csv_path = os.path.join(csv_dir, csv_filename, "config.csv")
            with open(csv_path, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 2:
                        key = row[0].strip()
                        value = row[1].strip()
                        # Try to convert numeric values
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        self.config[key] = value
        except Exception as e:
            print(f"Error loading config CSV: {e}")

    def _set_defaults(self):
        """Set default values if not already in config."""
        defaults = {
            'undeformed_diameter': 17,
            'pixel_size': 0.09,
            'voxel_depth': 1.0
        }
        for key, default in defaults.items():
            self.config.setdefault(key, default)

    def as_dict(self):
        """Return the configuration as a dictionary."""
        return self.config

    def __repr__(self):
        """Return a formatted string representation of the configuration."""
        return f"Config:\n{pprint.pformat(self.config, indent=2)}"
