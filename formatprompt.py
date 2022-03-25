class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def format_curl(params):
    nospace_prompt = params.prompt.replace(" ", "%20")
    curl_request = f"curl -X POST localhost:5000/getImage\?clip_input\=\"{nospace_prompt}\"\&folder_name\=\"{params.folder_name}\"\&session\=\"{params.session}\""
    return curl_request

params = Params(prompt="once upon a time there was a beautiful princess | princess with a crown of diamonds | Bryce3d; cinema4d toon and cell shader; VR perspective; finely intricate detail | 8k resolution; Unreal Engine VRay; ArtStation; CGSociety", folder_name="adventure", session="one")

print(format_curl(params))