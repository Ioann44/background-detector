## Install virtual environment
```bash
# install python if needed
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12-full -y
# create environment (recommended for all, not necessary though)
python3.12 -m venv python
# install packages
pip install -r requirements.txt
```

## Development
I use `black` formatter. Let's format the code before committing. You can do it with the following shortcut: `Ctrl+Alt+F`


## Utiities
```bash
# recursively resize images
python -m utils.resizer raw-data resized --info
# rename resized images (not necessary, but appreciated)
python -m utils.renamer
```

## Testing
For testing result model on your own images, you can try `gui_eval/web.py`. It will create a web server with a simple interface for uploading images and getting predictions