import face_recognition
import argparse
import imutils
import pickle
import cv2
from statistics import NormalDist
from progressbar import progressbar
from datetime import timedelta


def sample_size(confidence, popSize, marginError):
    z = NormalDist().inv_cdf((1 + confidence) / 2.0) ** 2
    a = 0.25 / (marginError ** 2)
    return round(z * (a / (1 + (z * (a / popSize)))), 2)


def output_frame(boxes, names, frame, r, args, fCount):
    print(f'[INFO] Saving frame {fCount} to output/{args["movie"][:-4]}')
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(
            frame, name, (left, y), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2
        )

    cv2.imwrite(f'output/{args["movie"][:-4]}/{fCount}.jpg', frame)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("movie", help="name with file extension of movie")
    args = vars(ap.parse_args())

    # load encoding information and video
    data = pickle.loads(open("encodings.pickle", "rb").read())
    video = cv2.VideoCapture(f'videos/{args["movie"]}')
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames to analyze: {totalFrames}")

    ss = sample_size(0.95, totalFrames, 0.1)
    print(f"[INFO] Estimated frames to sample: {ss}")

    modVal = round(totalFrames / ss, 0)

    # count total number of frames with matches
    mCount = 0

    for fCount in progressbar(range(totalFrames), redirect_stdout=True):
        grabbed, frame = video.read()

        if not grabbed:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameWidth = rgb.shape[1]

        if frameWidth > 750:
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])
        else:
            r = 1

        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        bMatch = 0

        for e in encodings:
            matches = face_recognition.compare_faces(data["encodings"], e)
            name = "not_nicolas_cage"

            if True in matches:
                matchedIndex = [i for i, b in enumerate(matches) if b]
                counts = {}

                for i in matchedIndex:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)
                bMatch = 1

            names.append(name)

        mCount += bMatch

        if fCount % modVal == 0:
            output_frame(boxes, names, frame, r, args, fCount)

    video.release()
    print(
        f"[INFO] {mCount} frames recognized, approximately {str(timedelta(seconds=round(mCount/24,0)))}"
    )
    with open("log.txt", "w") as oFile:
        oFile.write(str(mCount))
