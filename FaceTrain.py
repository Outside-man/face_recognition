#encoding:utf8

import pathlib
import face_recognition
import cv2
from FaceData import FaceData
import pickle

IMAGE_EXTENSIONS = [
    ".JPG",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png"
]
DIR = "pic"
def g_path(*args):
    path = pathlib.Path(DIR, *args)
    path.parent.mkdir(exist_ok=True)
    return path
def filter_dir(dir):
    yield from filter(lambda x: x.suffix in IMAGE_EXTENSIONS, g_path(dir).iterdir())

class FaceTrain:
    def __init__(self, DataArr):
        self.known_faces = []
        self.known_name = []
        for data in DataArr:
            t = []
            count = 1
            for file_path in filter_dir(data.path):
                try:
                    face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(file_path))
                    t.extend(face_encoding[0:min(1, len(face_encoding))])
                    print(data.name+str(count)+"导入")
                    count+=1
                except Exception as e:
                    print(data.name+str(count)+"导入失败")
                    continue
            if len(t) != 0:
                self.known_faces.append(t)
                self.known_name.append(data.name)

    def openCam(self):
        video_capture = cv2.VideoCapture(0)
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    count = 0
                    name = "unknown"
                    for known_face in self.known_faces:
                        matches = face_recognition.compare_faces(known_face, face_encoding, tolerance=0.45)

                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_name[count]
                            break
                        count += 1
                    face_names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    DataArr = []
    DataArr.append(FaceData("gakki/", "gakki"))
    face = FaceTrain(DataArr)
    with open('data.pk', 'wb') as f:
        data = pickle.dumps(face)
        pickle.dump(data, f)
    print("已经训练完毕")