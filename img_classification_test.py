from keras.models import load_model
import cv2
import numpy as np
import glob

def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size
    SCALAR_BLUE = (255.0, 0.0, 0.0)
    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 4.0
    fontThickness = 3

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function

# dimensions of our images
model=load_model('models/model3classesDogCatBirdAdam32.h5')
model.load_weights('models/checkpoints/weightsAdam32.h5')
#model.summary()

images= glob.glob("data/test/*.jpg")

for i in images:
    image = cv2.imread(i)
    imgr = cv2.resize(image, (150, 150))
    img = np.reshape(imgr, [1, 150, 150, 3])
    classes = model.predict(img)
    classification=np.argmax(classes)
    cv2.namedWindow('Prediction',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Prediction', 500,500)
    print(classes)
    print(classification)
    prediction=''
    if classification==1:
        prediction='Cat'
    elif classification==0:
        prediction='bird'
    elif classification==2:
        prediction=' Dog'

    writeResultOnImage(image,prediction)
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

