# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import base64
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from werkzeug.utils import secure_filename
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


data_dir='static/image'
input_size=299
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app = Flask(__name__) 
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg']) 

# transformations  
transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.ToTensor(),
])
def ml():
    model=torch.load('static/ml_model/model.pth',map_location='cpu')
    model.eval()
    classes=[]
    f=open("static/ml_model/classes.txt","r")
    for x in f:
        classes.append(x[:-1])
    f.close()
    to_pil = transforms.ToPILImage()
    data = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(data)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    for ii in range(len(images)):
        image = to_pil(images[ii])
        image_tensor = transform(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor) 
        input = input.to(device)
        output = model(input)
        index=output.data.cpu().numpy().argmax()
        result=str(classes[index])+" "
        f = open("static/ml_model/data.txt","r")
        for x, line in enumerate(f):
            if x == int(index):
                result+=str(line)
        f.close()
    return result


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload__from__mobile__app', methods=['POST'])
def api():
    if request.is_json:                
        content = request.get_json()
        File = content['base64']
        if File:
            file_object = base64.b64decode(File)
            filename = "image.jpeg"
            newFile = open (filename, "wb")
            newFile.write(file_object)
            os.rename(filename,"static/image/test_image/"+filename)
            result = ml()
            os.remove(os.path.join('static/image/test_image/', filename))
            print(result)
            return result
        else:
            return 'Error: No file found'
    else:
        return 'Error: Invalid file type'



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('static/image/test_image/', filename))
            result = ml()
            
            os.remove(os.path.join('static/image/test_image/', filename))
            return result
    return render_template('index.html')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_flex_quickstart]
