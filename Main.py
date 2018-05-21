import pickle

data = pickle.load(open("data.pk", "rb"))
face = pickle.loads(data)
face.openCam()