import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


video_capture=cv2.VideoCapture(0)

known_face_encodings=[]
known_face_names=[]

files=os.listdir("faces/")

for file in files:
    image=face_recognition.load_image_file(f"faces/{file}")
    image_encoding=face_recognition.face_encodings(image)[0]

    known_face_encodings.append(image_encoding)
    known_face_names.append(file.split(".")[0].upper())

students=known_face_names.copy()
students_present=[]

face_locations=[]
face_encodings=[]

now=datetime.now()

current_date=now.strftime("%Y-%m-%d")

f=open(f"{current_date}.csv","w+")
lnwriter=csv.writer(f)

choice=input("Enter 'A' for taking attendance.\nEnter 'R' for registering a new student.\n")


if choice.lower()=="a" and len(known_face_encodings)>0:
    print("Taking attendance of registered students.")
    while True:
        _,frame=video_capture.read()
        small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
            face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distance)

            if matches[best_match_index]:
                name=known_face_names[best_match_index]

                font=cv2.FONT_HERSHEY_COMPLEX
                fontscale=1
                fontcolor=(255,0,0)
                thickness=3
                linetype=2

                if name in known_face_names:
                    (text_width, text_height), baseline = cv2.getTextSize(f"{name} Present", font, fontscale, thickness)
                    image_height, image_width = image.shape[:2]
                    x = (image_width - text_width) // 2
                    cv2.putText(frame,f"{name} Present",(x,461),font,fontscale,fontcolor,thickness,linetype)

                    if name in students:
                        students.remove(name)
                        time=now.strftime("%H:%M:%S")
                        lnwriter.writerow([name,time])
            
            else:
                fontcolor=(0,0,255)
                cv2.putText(frame,"Student Not Found",(186,461),font,fontscale,fontcolor,thickness,linetype)
        cv2.imshow("Attendance",frame)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            print("Attendance marking completed.")
            break

elif choice.lower()=="a":
    print("No student have been enrolled for this class.")

elif choice.lower()=="r":
    print("Student Registration.")
    name=input("Enter the name of student :")
    if name.upper() in known_face_names:
        print("Student already registered.")
    else:
        known_face_names.append(name)
        students.append(name)
        while True:
            _,frame=video_capture.read()
            font=cv2.FONT_HERSHEY_COMPLEX
            fontscale=0.5
            fontcolor=(0,0,0)
            thickness=1
            linetype=1

            frame_copy=frame.copy()

            cv2.putText(frame,"Press 'S' to capture.",(3,455),font,fontscale,fontcolor,thickness,linetype)
            cv2.putText(frame,"Press 'Q' to exit.",(3,475),font,fontscale,fontcolor,thickness,linetype)
            cv2.imshow("Camera",frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): 
                cv2.imwrite(f"faces/{name}.jpg", frame_copy)
                print(f"Image saved as '{name}.jpg'")
            elif key == ord('q'):
                break

else:
    print("Invalid choice.")

video_capture.release()
cv2.destroyAllWindows()
f.close()