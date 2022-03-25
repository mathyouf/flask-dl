# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request
from diffusion_model import define_model

app = Flask(__name__)


@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    clip_input = request.args.get("clip_input")
    folder_name = request.args.get("folder_name")
    session = request.args.get("session")
    # Run the Network
    define_model(clip_input, folder_name, session)
    # Return after done running
    return


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    print("Enter the 🪄Imagination Machine🔮")
    app.run(host='0.0.0.0')