
## Project Overview
image_similarity_recommendation_system

## Dataset
Due to GitHub file size limitations, the dataset is hosted externally(google drive).
📥 Dataset Download Link:  
https://drive.google.com/drive/folders/1MisrfoAR6OT6cGsVuYhCtYFGIg-i49uS?usp=drive_link

## Pretrained Model
The trained ML model is also hosted externally(google drive).
📥 Model Download Link:  
https://drive.google.com/file/d/1BPPUieypsDUW6-hjzluHeXPmZnZ9eMRy/view?usp=drive_link

## Folder Structure
project/
│── Dataset/                       ##Place the downloaded dataset here
│── Pretrained Model/              ## place downloaded model here
│── requirements.txt               ##download the packages and libraries to execute our project
│── app.py                         ##It is a web application interface
│── model.py 
│── build_db                       ##Database connection 
	    │── db
		      │── embedding.pt         #It is a pytorch file which is used to convert the images into vector representation for further purpose
		      │──paths.pt              #It is used to access the location of the vector faster
│── README.md


# Steps to execute the Project
1. Download the dataset and pretrained model from the drive
2. Install all the packages from the requirements.txt file
3. Create a folder and save the model.py, build.py.
4. execute the interface application which is "app.py" with "streamlit app.py" command
5. when you execute it, from "model.py" it creates a "build_db folder" and its files we can see them in "folder structure".

# Video Explanation Link
📥https://drive.google.com/file/d/1WyHbUI-nyy0uYoJ4Qsui2avuRcbC4Gbn/view?usp=drive_link
