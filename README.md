# The Imagination Machine API 

## Run these in AWS, a machine with a GPU
Setup using `bash ./setup`

Run using `python server.py`

## Run these from anywhere (eventually, our Oculus Quest from Unity)
Replace `localhost` with `Public IPv4 address` for AWS EC2 hosted servers
Test using `curl -X POST localhost:5000/getImage\?clip_input\="flower%20petal"\&folder_name\="first_folder"\&session\="first"&cutn=32&clip_guidance_scale=50000&tv_scale=80000`

`outputFile.png` will be the generated image, and will be saved to the local directroy you are calling `curl` from.

(Python should be version >3.6)