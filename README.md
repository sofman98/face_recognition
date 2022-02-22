# face_recognition
Face Recognition with anti-spoofing using only RGB Cameras

Recognize multiple people (ideal for a security camera)  by running
```
$ python recognize.py
```
![alt text](https://github.com/sofman98/face_recognition/blob/main/demo/demo_multiple_people.gif?raw=true)

Or recognize one single person with anti-spoofing activated (ideal for phone unlocking and emloyee attendance punching machine)
```
$ python recognize_with_antispoof.py
```
![alt text](https://github.com/sofman98/face_recognition/blob/main/demo/demo_antispoof.gif?raw=true)

Don't forget to add the faces in the "faces" folder and modify "known_faces.py" accordingly!

Special thanks to @Minivision_AI and @ageitgey.
