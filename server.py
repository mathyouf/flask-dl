# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request
from diffusion_model import define_model
import numpy as np
import cv2

class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

app = Flask(__name__)
params = Params()

@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    rf = request.form
    nparr = np.fromtring(request.data, np.uint8)
    params.init_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Get text input
    params.clip_input = rf.get("clip_input", "A picture of an industrial design rendering. CGTrader Blender UnrealEngine ArtStation")
    params.folder_name = rf.get("folder_name")
    params.session = rf.get("session")
    params.cutn = rf.get("cutn", 32)
    params.clip_guidance_scale = rf.get("clip_guidance_scale", 50000)
    params.tv_scale = rf.get("tv_scale", 50000)
    params.img_size = rf.get("img_size", 512)
    params.num_steps = rf.get("num_steps", 500)

    # Run the Network
    define_model(clip_input=params.clip_input, folder_name=params.folder_name, session=params.session, cutn=params.cutn, clip_guidance_scale=params.clip_guidance_scale, tv_scale=params.tv_scale, img_size=params.img_size, num_steps=params.num_steps)
    # Make into video
    makeMp4command=['bash', './makeMP4', params.session, params.folder_name]
    subprocess.call(makeMp4command)
    # Return after done running
    return "All done."


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0', threaded=True)