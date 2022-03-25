# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request
from diffusion_model import define_model

def runNetwork(clip_input):
    send_file(clip_input)


app = Flask(__name__)


@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    print(request.args)
    clip_input = request.args.get('clip_input')
    # Run the Network
    runNetwork(clip_input)
    # This file is generated at the end of do_run.cond_nf
    filename = "progress_00000.png"
    # Return image
    return send_file(filename, mimetype="image/png")


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0')