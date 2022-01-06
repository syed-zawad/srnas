from PIL import Image

#read the image
im = Image.open("sample-image.png")

#image size
size=(89,)
#resize image
out = im.resize(size)
#save resized image
out.save('resize-output.png')
