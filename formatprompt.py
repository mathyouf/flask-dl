class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def format_curl(params):
    nospace_prompt = params.prompt.replace(" ", "%20")
    curl_request = f"curl -X POST localhost:5000/getImage\?clip_input\=\"{nospace_prompt}\"\&folder_name\=\"{params.folder_name}\"\&session\=\"{params.session}\""
    return curl_request

params = Params(prompt="her soul was a dandelion, carried to all corners of the earth by the wind | fractal composition with a central sunflower | geometric mandala of swirling color; snow melting, crystals evaporating into a steam, into nothing | Bryce3d; cinema4d; VR perspective; finely intricate detail | 8k resolution; Unreal Engine VRay; 3d smooth; ArtStation; CGSociety", folder_name="adventure", session="one")

print(format_curl(params))