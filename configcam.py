# configcam.py Camera Settings. These are imported by strmcam.py

CAMERA = "pilibcam"    # valid values usbcam, rtspcam, pilibcam, pilegcam
USBCAM_SRC = 0         # Device number of USB connection usually 0, 1, 2, Etc
RTSPCAM_SRC = "rtsp://user:password@IP:554/url"  # Set per IP Cam Docs and config

# Camera Image Stream Settings
IM_SIZE = (1800, 1000)   # Image resolution width, height pixels
IM_VFLIP = True       # True enables flipping image vertically
IM_HFLIP = True       # True enables flipping image horizonally
IM_ROTATION = 0        # Rotate camera image valid values are 0, 90, 180, 270
IM_FRAMERATE = 100      # Legacy Picamera Framerate
