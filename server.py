# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request
from diffusion_model import define_model

class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

app = Flask(__name__)
params = Params()

@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    print("endpoint hit:)")
    params.clip_input = request.args.get("clip_input")
    params.folder_name = request.args.get("folder_name")
    params.session = request.args.get("session")
    try:
        params.cutn = request.args.get("cutn")
        params.clip_guidance_scale = request.args.get("clip_guidance_scale")
        params.tv_scale = request.args.get("tv_scale")
        params.num_steps = request.args.get("num_steps")
        params.img_size = request.args.get("img_size")
    except:
        print("no params given for cutn, clip_guidance_scale, tv_scale, num_steps, and/or img_size")
    
    print("PARAMS:\n", params)
    # Run the Network
    define_model(params.clip_input, params.folder_name, params.session, params.cutn, params.clip_guidance_scale, params.tv_scale, params.num_steps, params.img_size)
    # Return after done running
    return "All done."


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0')