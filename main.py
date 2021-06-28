# import the necessary packages
from flask import Flask, render_template, Response
from detection_and_recognition import VideoCamera

app = Flask(__name__)
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/about.html')
def about():
    # rendering webpage
    return render_template('about.html')
@app.route('/index.html')
def index2():
    # rendering webpage
    return render_template('index.html')
@app.route('/video')
def video():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)