import cv2
import dlib
import numpy as np
import os
import imutils

# Face image directories
dirPath = os.path.dirname(os.path.realpath(__file__))
faceImagePath = dirPath + "\\images\\face\\"

# Face predictor data path
p = "data/shape_predictor_68_face_landmarks.dat"

# Mask colours
color_blue = (254, 207, 110)
color_cyan = (255, 200, 0)
color_black = (0, 0, 0)

# Get user input for mask type and mask colour
maskColour = input("Please select the choice of mask color\nEnter 1 for blue\nEnter 2 for black:\n")
maskColour = int(maskColour)

if maskColour == 1:
    maskColour = color_blue
    print('You selected mask color = blue')
elif maskColour == 2:
    maskColour = color_black
    print('You selected mask color = black')
else:
    print("Invalid selection, please select again.")
    input("Please select the choice of mask color\nEnter 1 for blue\nEnter 2 for black :\n")

maskCoverageType = input(
    "Please enter choice of mask type coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")
maskCoverageType = int(maskCoverageType)

if maskCoverageType == 1:
    print(f'You chosen wide, high coverage mask')
elif maskCoverageType == 2:
    print(f'You chosen wide, medium coverage mask')
elif maskCoverageType == 3:
    print(f'You chosen wide, low coverage mask')
else:
    print("invalid selection, please select again.")
    input("Please enter choice of mask type coverage \nEnter 1 for high \nEnter 2 for medium \nEnter 3 for low :\n")

for filename in os.listdir(faceImagePath):

    # Load image -> resize -> convert to grayscale
    img = cv2.imread("images/face/" + filename)
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces and store its list of bounding rectangles coordinates
    faces = detector(gray, 1)
    print(faces)
    print("Number of faces detected: ", len(faces))



    # Initialize dlib's shape predictor
    predictor = dlib.shape_predictor(p)

    # Get the shape using the predictor

    for face in faces:
        landmarks = predictor(gray, face)

        points = []
        for i in range(1, 16):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            points.append(point)

        # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
        maskHigh = [((landmarks.part(42).x), (landmarks.part(15).y)),
                  ((landmarks.part(27).x), (landmarks.part(27).y)),
                  ((landmarks.part(39).x), (landmarks.part(1).y))]

        # Coordinates for the additional point for wide, medium coverage mask - in sequence
        maskMed = [((landmarks.part(29).x), (landmarks.part(29).y))]

        # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
        maskLow = [((landmarks.part(35).x), (landmarks.part(35).y)),
                  ((landmarks.part(34).x), (landmarks.part(34).y)),
                  ((landmarks.part(33).x), (landmarks.part(33).y)),
                  ((landmarks.part(32).x), (landmarks.part(32).y)),
                  ((landmarks.part(31).x), (landmarks.part(31).y))]

        fmaskHigh = points + maskHigh
        fmaskMed = points + maskMed
        fmaskLow = points + maskLow

        fmaskHigh = np.array(fmaskHigh, dtype=np.int32)
        fmaskMed = np.array(fmaskMed, dtype=np.int32)
        fmaskLow = np.array(fmaskLow, dtype=np.int32)

        mask_type = {1: fmaskHigh, 2: fmaskMed, 3: fmaskLow}
        mask_type[maskCoverageType]

        # change parameter [mask_type] and color_type for various combination
        img2 = cv2.polylines(img, [mask_type[maskCoverageType]], True, maskColour, thickness=2, lineType=cv2.LINE_8)

        # Using Python OpenCV – cv2.fillPoly() method to fill mask
        # change parameter [mask_type] and color_type for various combination
        img3 = cv2.fillPoly(img2, [mask_type[maskCoverageType]], maskColour, lineType=cv2.LINE_AA)


    # Save the output file for testing
    outputNameofImage = "output/mask_dlib/imagetest.jpg"
    print("Saving output image to", outputNameofImage)
    cv2.imwrite(outputNameofImage, img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
