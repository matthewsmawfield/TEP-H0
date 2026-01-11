
import requests

url = "http://atlas.obs-hp.fr/hyperleda/ledacat.cgi"
targets = ['NGC 7541', 'NGC 1448']

print(f"--- Querying Fields Individually on {url} ---")

for target in targets:
    print(f"\nTARGET: {target}")
    
    # Query vdis only
    params_vdis = {'o': target, 'd': 'vdis', 'a': 't'}
    try:
        resp = requests.get(url, params=params_vdis, timeout=10)
        lines = [l.strip() for l in resp.text.split('\n') if l.strip() and not l.startswith('#') and not l.startswith('<')]
        if lines:
            print(f"  vdis raw: '{lines[0]}'")
        else:
            print(f"  vdis raw: <empty>")
    except Exception as e:
        print(f"  vdis error: {e}")

    # Query vrot only
    params_vrot = {'o': target, 'd': 'vrot', 'a': 't'}
    try:
        resp = requests.get(url, params=params_vrot, timeout=10)
        lines = [l.strip() for l in resp.text.split('\n') if l.strip() and not l.startswith('#') and not l.startswith('<')]
        if lines:
            print(f"  vrot raw: '{lines[0]}'")
        else:
            print(f"  vrot raw: <empty>")
    except Exception as e:
        print(f"  vrot error: {e}")
