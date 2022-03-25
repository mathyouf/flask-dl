# The Imagination Machine API 

## Run these in AWS, a machine with a GPU
Setup using `bash ./setup`

Run using `python server.py`

## Run these from anywhere (eventually, our Oculus Quest from Unity)
Replace `localhost` with `Public IPv4 address` for AWS EC2 hosted servers
Test using `curl -X POST localhost:5000/getImage\?clip_input\="hi" -o outputFile.png`

`outputFile.png` will be the generated image, and will be saved to the local directroy you are calling `curl` from.

(Python should be version >3.6)