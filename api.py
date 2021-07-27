from flask import Flask,render_template,redirect
import os
from flask.globals import request
from flask.helpers import url_for
from markupsafe import escape
import load_model
import matplotlib.pyplot as plt
from PIL import Image
import torch


upload_folder='uploads'
allow_ext=set(['jpg','png','jpeg','jfif'])

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
size=224

transform=load_model.BaseTransform(size,mean,std)
classifier=['cat','dog']

app=Flask(__name__)
app.config['UPLOAD_FOLDER']=upload_folder


def allow_filename(filename):
    return '.' in filename and filename.split('.')[-1] in allow_ext


@app.route('/predict/<img_path>')
def predict(img_path='static\img\dog7.jpeg'):
    model=load_model.make_model()
    img=Image.open(img_path)
    img_show=load_model.transform_umnorm(img)
    plt.imshow(img_show.numpy().transpose(1,2,0))
    img_pred=transform(img)
    img_pred=img_pred.unsqueeze(0)
    out=model(img_pred)
    print(out)
    pred=torch.argmax(out,1)
    title=classifier[pred.item()]
    print(title)
    return title

@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST' and 'image' in request.files:
        f=request.files['image']
        img_path=os.path.join(app.config['UPLOAD_FOLDER'],f.filename)
        f.save(img_path)
        return redirect(url_for('predict',img_path=img_path))
    return render_template('upload_image.html')    

