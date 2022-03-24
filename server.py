# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, send_file, request

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


def runNetwork(clip_input):
    filename = 'sample_images/horror.png'
    return filename
    pass


@app.route('/getImage', methods=["POST"])
def VisualImaginationMachine():
    # Get text input
    clip_input = request.args.get('clip_input')
    # Run the Network
    filename = runNetwork(clip_input)
    # Return image
    return send_file(filename, mimetype="image/png")

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()