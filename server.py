# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request
from diffusion_model import define_model

app = Flask(__name__)


@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    print("endpoint hit:)")
    clip_input = request.args.get("clip_input")
    folder_name = request.args.get("folder_name")
    session = request.args.get("session")
    try:
        cutn = request.args.get("cutn")
        clip_guidance_scale = request.args.get("clip_guidance_scale")
        tv_scale = request.args.get("tv_scale")
        num_steps = request.args.get("num_steps")
        img_size = request.args.get("img_size")
    except:
        print("no params given for cutn, clip_guidance_scale, tv_scale, num_steps, and/or img_size")
    print("clip_input", clip_input, "folder_name", folder_name, "session", session)
    # Run the Network
    define_model(clip_input, folder_name, session, cutn, clip_guidance_scale, tv_scale, num_steps, img_size)
    # Return after done running
    return "All done."


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0')