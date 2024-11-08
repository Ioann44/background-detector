from bing_image_downloader import downloader
downloader.download('fantasy background', limit=1000,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=3600, verbose=True)