class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def format_curl(params):
    nospace_prompt = params.prompt.replace(" ", "%20")
    curl_request = f"curl -X POST localhost:5000/getImage\?clip_input\=\"{nospace_prompt}\"\&folder_name\=\"{params.folder_name}\"\&session\=\"{params.session}\""
    return curl_request

params = Params(prompt="bright lattices of logic unfolding across that colourless void | intricate recursion portal of virtual matrix realm to the internet | Bryce3d; cinema4d toon and cell shader; VR perspective | 8k resolution; Unreal Engine VRay; ArtStation; CGSociety", folder_name="adventure", session="one")

print(format_curl(params))