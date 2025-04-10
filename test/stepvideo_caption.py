import pickle
import requests


def call_api(url, api, port=8080, samples=None, **kwargs):
    full_url = f"http://{url}:{port}/{api}-api"

    if api == 'vae':
        data = {"samples": samples}
    elif api == 'caption':
        data = {"prompts": samples}
    else:
        raise Exception(f"Not supported api: {api}...")

    data_bytes = pickle.dumps(data)
    response = requests.get(full_url, data=data_bytes, timeout=12000)
    response_data = pickle.loads(response.content)

    return response_data