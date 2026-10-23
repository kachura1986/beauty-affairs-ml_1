import json
import os


class FileIO:
    """
    A class for file I/O operations.
    """

    @classmethod
    def load_file(cls, path: str) -> tuple[bytes, str, str]:
        """
        Load a local file system and return the file content as bytes, name, and extension.

        Args:
            path: The path of a local file path.

        Returns:
            A tuple containing the file content as bytes, name, and extension.
        """
        file_name, file_extension = os.path.splitext(os.path.basename(path))

        with open(path, 'rb') as f:
            file_content = f.read()

        return file_content, file_name, file_extension

    @classmethod
    def save_file(cls, path: str, content: bytes) -> None:
        """
        Save the content to the specified file path.

        Args:
            path: The path to save the content to.
            content: The bytes content to be saved.
        """
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        with open(path, 'wb') as f:
            f.write(content)

    @classmethod
    def load_json(cls, path: str) -> tuple[dict | list | str, str, str]:
        """
        Load the JSON data from the specified path.

        Args:
            path: The path to the JSON file.

        Returns:
            A tuple containing the JSON data as a dictionary or list or string, file name, and file extension.
        """
        json_bytes, file_name, file_extension = FileIO.load_file(path)
        json_content = json.loads(json_bytes.decode('utf-8'))

        return json_content, file_name, file_extension

    @classmethod
    def save_json(cls, data: str | dict, path: str) -> None:
        """
        Saves data to a JSON file, automatically handling stringified JSON.

        Args:
            data (str | dict): Data to be saved. If a string is provided, it must be a valid JSON string.
            path (str): Path to the JSON file where data will be saved.

        Raises:
            ValueError: If the provided string is not a valid JSON format.
        """
        # Convert string JSON to dictionary if needed
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("The provided string is not a valid JSON format.")
        else:
            parsed_data = data

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save data to the specified file
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(parsed_data, file, indent=4, ensure_ascii=False)
