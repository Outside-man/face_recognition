import pickle
from FaceTrain import FaceTrain
data = pickle.load(open("data.pk", "rb"))
face = pickle.loads(data)
face.openCam()
