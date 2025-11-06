import base64, requests

def b64(path):
    with open(path,'rb') as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

payload = {"imageBase64": b64("sample.png"), "template": "pta_v1"}
r = requests.post("http://127.0.0.1:5001/ocr/pta", json=payload, timeout=120)
print(r.status_code)
print(r.json())
