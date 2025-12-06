# scripts/test_splat_physics.py
from mujoco import MjModel, MjData
from mujoco.viewer import launch
import time
import numpy as np

MODEL_XML = "meshobjects/splat_from_meshextract/splat_from_meshextract.xml"  # update path

model = MjModel.from_xml_path(MODEL_XML)
data = MjData(model)

# helper to get contact info
def print_contacts():
    # data.ncon : number of contacts
    ncon = data.ncon
    if ncon == 0:
        print("No contacts")
        return
    print(f"Contacts: {ncon}")
    # In MuJoCo v2+ Python API, contacts are in data.contact, an array-like of
    # objects with fields (id, geom1, geom2, ...). We'll print geom indices and force.
    # If your mujoco Python API version exposes different attributes, adjust accordingly.
    for i in range(ncon):
        c = data.contact[i]
        print(f" contact {i}: geom1={c.geom1}, geom2={c.geom2}, dist={c.dist:.6f}, frame={c.frame}, pos={c.pos}, frameforce={c.frameforce}")

def run_sim(steps=2000):
    with launch(model, data) as viewer:
        for i in range(steps):
            # step simulation
            data.step()
            # print contact info every 10 frames
            if i % 10 == 0:
                print_contacts()
            viewer.render()
            time.sleep(0.002)

if __name__ == "__main__":
    run_sim()