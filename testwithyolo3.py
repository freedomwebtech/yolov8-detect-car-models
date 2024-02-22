import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import requests
from bs4 import BeautifulSoup
import glob
import cvzone
model = YOLO('best.pt')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

def get_car_details(car_company, car_model):
    # URL of the website with placeholders for company and model
    url = f"https://www.cartrade.com/{car_company.lower()}-cars/{car_model.lower()}/"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all <p> tags with a specific class
        p_elements = soup.find_all('p', class_='specTitle mt-0 mb-0 mr-0 font-weight-normal')
        
        # Find all <td> tags with a specific class
        td_elements = soup.find_all('td', class_='specData')
        
        # Extract the text from each <p> tag and <td> tag
        details_text = ""
        for p_element, td_element in zip(p_elements, td_elements):
            p_text = p_element.text.strip()
            td_text = td_element.text.strip()
            # Check if the line contains the word "price" and remove "???" sign
            if "price" in p_text.lower() and td_text == "???":
                td_text = ""
            details_text += f"{p_text}: {td_text}\n"
        
        # Return the car details as plain text
        return details_text
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"

# Read the image
path = r'C:\Users\freed\Downloads\websitecardekho\carimg\*.*'
images = glob.glob(path)

for file in glob.glob(path):
    img = cv2.imread(file)
    img = cv2.resize(img, (1020, 500))
    results = model.predict(img)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        print(c)
        # Split the detected result into car company and model
        parts = c.rsplit('-', 2)  # Split based on last occurrence of '-'
        car_company = parts[0]
        car_model = '-'.join(parts[1:])  # Re-join the model part if it contains multiple '-'
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(img, f'{car_company} {car_model}', (x1, y1), 1, 1)

        # Get car details
        car_details = get_car_details(car_company, car_model)
        
        # Split car details into lines
        details_lines = car_details.split('\n')
        
        # Create a separate window for car details
        details_window = np.ones((500,700, 3), np.uint8) * 255
        
        # Draw car details on the window line by line
        y_offset = 30
        for line in details_lines:
#            cv2.putText(details_window, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0,255), 2)
            cvzone.putTextRect(details_window, line, (30, y_offset), 1, 1)

            y_offset += 30  # Increment y_offset for the next line
            
        # Display the car details window
        cv2.imshow("Car Details", details_window)
        
        # Display the main image
        cv2.imshow("img", img)
        
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
